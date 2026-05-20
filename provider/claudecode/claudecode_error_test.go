package claudecode

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

func TestClassifyProviderError_AuthFailed(t *testing.T) {
	waitErr := errors.New("exit status 1")
	tests := []struct {
		name   string
		stderr string
	}{
		{"not logged in", "Not logged in. Please run /login."},
		{"authentication_failed", `{"type":"result","is_api_error":true,"error":"authentication_failed"}`},
		{"run /login", "Error: please run /login to authenticate"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pe := classifyProviderError(tt.stderr, waitErr)
			if pe.Kind != bridle.ProviderErrorAuthFailed {
				t.Errorf("Kind = %q; want %q", pe.Kind, bridle.ProviderErrorAuthFailed)
			}
			if pe.Err != waitErr {
				t.Error("underlying error not preserved")
			}
			if pe.Message == "" {
				t.Error("Message is empty")
			}
			// The human-readable Message must not contain the raw exit code.
			if strings.Contains(pe.Message, "exit status") {
				t.Errorf("Message contains raw exit code: %q", pe.Message)
			}
			t.Logf("auth_failed message: %s", pe.Error())
		})
	}
}

func TestClassifyProviderError_RateLimit(t *testing.T) {
	waitErr := errors.New("exit status 1")
	pe := classifyProviderError("rate_limit: too many requests", waitErr)
	if pe.Kind != bridle.ProviderErrorRateLimit {
		t.Errorf("Kind = %q; want %q", pe.Kind, bridle.ProviderErrorRateLimit)
	}
	t.Logf("rate_limit message: %s", pe.Error())
}

func TestClassifyProviderError_ServerError(t *testing.T) {
	waitErr := errors.New("exit status 1")
	tests := []struct {
		name   string
		stderr string
	}{
		{"server_error", "server_error: internal error"},
		{"internal server error", "internal server error (500)"},
		{"overloaded", "overloaded_error: try again later"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pe := classifyProviderError(tt.stderr, waitErr)
			if pe.Kind != bridle.ProviderErrorServerError {
				t.Errorf("Kind = %q; want %q", pe.Kind, bridle.ProviderErrorServerError)
			}
		})
	}
}

func TestClassifyProviderError_NetworkError(t *testing.T) {
	waitErr := errors.New("exit status 1")
	tests := []struct {
		name   string
		stderr string
	}{
		{"connection refused", "Error: dial tcp 127.0.0.1:443: connection refused"},
		{"no route to host", "Error: no route to host"},
		{"connection reset", "Error: read tcp ... connection reset by peer"},
		{"eof", "Error: unexpected EOF"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pe := classifyProviderError(tt.stderr, waitErr)
			if pe.Kind != bridle.ProviderErrorNetworkError {
				t.Errorf("Kind = %q; want %q", pe.Kind, bridle.ProviderErrorNetworkError)
			}
		})
	}
}

func TestClassifyProviderError_Timeout(t *testing.T) {
	waitErr := errors.New("exit status 1")
	tests := []struct {
		name   string
		stderr string
	}{
		{"timeout", "Error: context deadline exceeded (timeout)"},
		{"deadline exceeded", "Error: deadline exceeded"},
		{"timed out", "Error: request timed out after 30s"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pe := classifyProviderError(tt.stderr, waitErr)
			if pe.Kind != bridle.ProviderErrorTimeout {
				t.Errorf("Kind = %q; want %q", pe.Kind, bridle.ProviderErrorTimeout)
			}
		})
	}
}

func TestClassifyProviderError_TLSError(t *testing.T) {
	waitErr := errors.New("exit status 1")
	tests := []struct {
		name   string
		stderr string
	}{
		{"certificate", "Error: x509: certificate signed by unknown authority"},
		{"ssl", "Error: SSL handshake failed"},
		{"tls", "Error: tls: protocol version not supported"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pe := classifyProviderError(tt.stderr, waitErr)
			if pe.Kind != bridle.ProviderErrorTLSError {
				t.Errorf("Kind = %q; want %q", pe.Kind, bridle.ProviderErrorTLSError)
			}
		})
	}
}

func TestClassifyProviderError_GenericFallback(t *testing.T) {

	waitErr := errors.New("exit status 2")
	pe := classifyProviderError("some unexpected stderr output", waitErr)
	if pe.Kind != "subprocess_exit" {
		t.Errorf("Kind = %q; want subprocess_exit", pe.Kind)
	}
	if pe.Err != waitErr {
		t.Error("underlying error not preserved")
	}
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name string
		pe   *bridle.ProviderError
		want bool
	}{
		{"rate_limit", &bridle.ProviderError{Kind: bridle.ProviderErrorRateLimit, Message: "rate limited"}, true},
		{"server_error", &bridle.ProviderError{Kind: bridle.ProviderErrorServerError, Message: "server error"}, true},
		{"network_error", &bridle.ProviderError{Kind: bridle.ProviderErrorNetworkError, Message: "network error"}, true},
		{"timeout", &bridle.ProviderError{Kind: bridle.ProviderErrorTimeout, Message: "timed out"}, true},
		{"auth_failed", &bridle.ProviderError{Kind: bridle.ProviderErrorAuthFailed, Message: "auth failed"}, false},
		{"tls_error", &bridle.ProviderError{Kind: bridle.ProviderErrorTLSError, Message: "tls error"}, false},
		{"subprocess_exit", &bridle.ProviderError{Kind: "subprocess_exit", Message: "generic exit"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isRetryable(tt.pe)
			if got != tt.want {
				t.Errorf("isRetryable(%q) = %v; want %v", tt.pe.Kind, got, tt.want)
			}
		})
	}

	// Non-ProviderError should not be retryable.
	if isRetryable(errors.New("some other error")) {
		t.Error("isRetryable on plain error should be false")
	}
}

func TestIsSessionIDInUseErr(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"exact match", errors.New("Session ID abc123 is already in use"), true},
		{"ANSI prefix", errors.New("\x1b[31mSession ID xyz is already in use\x1b[0m"), true},
		{"nil", nil, false},
		{"unrelated", errors.New("connection refused"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isSessionIDInUseErr(tt.err); got != tt.want {
				t.Errorf("isSessionIDInUseErr(%v) = %v; want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestIsAPIError(t *testing.T) {
	tests := []struct {
		name  string
		event map[string]json.RawMessage
		want  bool
	}{
		{
			name:  "snake_case true",
			event: map[string]json.RawMessage{"is_api_error": json.RawMessage("true")},
			want:  true,
		},
		{
			name:  "camelCase true",
			event: map[string]json.RawMessage{"isApiErrorMessage": json.RawMessage("true")},
			want:  true,
		},
		{
			name:  "snake_case false",
			event: map[string]json.RawMessage{"is_api_error": json.RawMessage("false")},
			want:  false,
		},
		{
			name:  "neither field",
			event: map[string]json.RawMessage{"type": json.RawMessage(`"assistant"`)},
			want:  false,
		},
		{
			name:  "camelCase false",
			event: map[string]json.RawMessage{"isApiErrorMessage": json.RawMessage("false")},
			want:  false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isAPIError(tt.event)
			if got != tt.want {
				t.Errorf("isAPIError = %v; want %v", got, tt.want)
			}
		})
	}
}

// TestParseStream_APIErrorEvent verifies that parseStream emits a TurnError
// when the stream contains an event with is_api_error=true, and that the
// accumulated text content is preserved for the caller to classify.
func TestParseStream_APIErrorEvent(t *testing.T) {
	// Simulate the auth-failure stream: synthetic assistant message
	// followed by an API error event, no result event.
	stream := strings.NewReader(strings.Join([]string{
		`{"type":"assistant","message":{"content":[{"type":"text","text":"Not logged in. Please run /login."}]}}`,
		`{"type":"result","subtype":"error_during_execution","is_api_error":true,"error":"authentication_failed"}`,
	}, "\n"))

	sink := &fake.SliceEventSink{}
	result, err := parseStream(stream, sink)
	if err != nil {
		t.Fatalf("parseStream unexpected error: %v", err)
	}

	// parseStream should surface the assistant text.
	if !strings.Contains(result.FinalText, "Not logged in") {
		t.Errorf("FinalText = %q; expected it to contain 'Not logged in'", result.FinalText)
	}

	// Should have emitted a TurnError for the API error.
	var foundAPIError bool
	for _, e := range sink.Events {
		if te, ok := e.(bridle.TurnError); ok && te.Stage == "provider_api_error" {
			foundAPIError = true
			t.Logf("TurnError captured: %v", te.Err)
		}
	}
	if !foundAPIError {
		t.Error("expected TurnError with stage=provider_api_error to be emitted")
	}
	t.Logf("events emitted: %d", len(sink.Events))
}

// --- NEX-241: subprocess diagnostic helpers ---

func TestExcerpt_Truncation(t *testing.T) {
	tests := []struct {
		name  string
		line  []byte
		max   int
		want  string
	}{
		{"shorter than max", []byte("hello"), 10, "hello"},
		{"equal to max", []byte("hello"), 5, "hello"},
		{"longer than max", []byte("hello world"), 5, "hello"},
		{"empty", []byte(""), 5, ""},
		{"max zero", []byte("hello"), 0, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := excerpt(tt.line, tt.max)
			if got != tt.want {
				t.Errorf("excerpt(%q, %d) = %q; want %q", tt.line, tt.max, got, tt.want)
			}
		})
	}
}

func TestRotateLogs_ShiftsFiles(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "claude-stderr.log")

	// Write a file large enough to trigger rotation (the rotation gate
	// checks size via Stat; we'll fake it by writing a small file and
	// testing the shift logic directly instead).
	// Actually, let's just test the shift by creating files manually
	// and calling rotateLogs on a small file — the size gate will skip.
	// Test the shifting by creating a large-enough dummy file.
	large := make([]byte, 10*1024*1024+1)
	if err := os.WriteFile(base, large, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(base+".1", []byte("previous"), 0600); err != nil {
		t.Fatal(err)
	}

	rotateLogs(base)

	// After rotation: base should NOT exist (it was rotated to .1),
	// .1 should contain what base had, .2 should contain what .1 had.
	if _, err := os.Stat(base); err == nil {
		t.Error("base file should have been rotated away")
	}
	data1, err := os.ReadFile(base + ".1")
	if err != nil {
		t.Fatal(err)
	}
	if len(data1) != len(large) {
		t.Errorf(".1 size = %d; want %d", len(data1), len(large))
	}
	data2, err := os.ReadFile(base + ".2")
	if err != nil {
		t.Fatal(err)
	}
	if string(data2) != "previous" {
		t.Errorf(".2 content = %q; want %q", string(data2), "previous")
	}

	// .3 should not exist (retain=3 keeps current + .1 + .2).
	if _, err := os.Stat(base + ".3"); err == nil {
		t.Error(".3 should not exist (retain=3)")
	}
}

func TestWriteStderrLog_EmptyNoop(t *testing.T) {
	dir := t.TempDir()
	// Empty stderr
	if got := writeStderrLog(dir, ""); got != "" {
		t.Errorf("empty stderr should return empty path, got %q", got)
	}
	// Empty cwd
	if got := writeStderrLog("", "some output"); got != "" {
		t.Errorf("empty cwd should return empty path, got %q", got)
	}
}

func TestWriteStderrLog_WritesAndReturnsPath(t *testing.T) {
	dir := t.TempDir()
	got := writeStderrLog(dir, "claude stderr output line 1\nline 2")
	expectedPath := filepath.Join(dir, "tmp", "claude-stderr.log")
	if got != expectedPath {
		t.Fatalf("writeStderrLog = %q; want %q", got, expectedPath)
	}

	content, err := os.ReadFile(expectedPath)
	if err != nil {
		t.Fatal(err)
	}
	s := string(content)
	if !strings.Contains(s, "claude stderr output line 1") {
		t.Errorf("stderr log missing content: %s", s)
	}
	if !strings.Contains(s, "line 2") {
		t.Errorf("stderr log missing second line: %s", s)
	}
	// Should include RFC3339 timestamp separator.
	if !strings.Contains(s, "===") {
		t.Errorf("stderr log missing timestamp separator: %s", s)
	}
}

func TestWriteStderrLog_CreatesTmpDir(t *testing.T) {
	dir := t.TempDir()
	// tmp/ subdirectory doesn't exist yet.
	tmpDir := filepath.Join(dir, "tmp")
	if _, err := os.Stat(tmpDir); err == nil {
		t.Skip("tmp already exists")
	}
	got := writeStderrLog(dir, "hello")
	if got == "" {
		t.Fatal("writeStderrLog returned empty")
	}
	fi, err := os.Stat(tmpDir)
	if err != nil {
		t.Fatalf("tmp dir not created: %v", err)
	}
	if !fi.IsDir() {
		t.Error("tmp is not a directory")
	}
}

func TestDiagnosticsSuffix_AllComponents(t *testing.T) {
	// Build a parseResult with known fields.
	presult := parseResult{
		lastEventType:    "assistant",
		lastEventExcerpt: `{"type":"assistant","message":{"content":[{"type":"text","text":"some model output that goes on for quite a while and ends up truncated here because we only keep the excerpt of the line which captures 200 characters which is enough to diagnose what was being emitted when the subprocess died..."}]}}`,
	}
	runtime := 2500 * time.Millisecond
	stderrLogPath := "/tmp/aspect/claude-stderr.log"

	suffix := diagnosticsSuffix(nil, presult, runtime, stderrLogPath)

	// Each component should appear.
	wants := []string{
		"runtime=2.5s",
		"last_event=assistant",
		"last_event_excerpt=",
		"stderr_log=/tmp/aspect/claude-stderr.log",
	}
	for _, w := range wants {
		if !strings.Contains(suffix, w) {
			t.Errorf("diagnosticsSuffix missing %q in: %s", w, suffix)
		}
	}

	// With nil error, no exit_code or signal should appear.
	if strings.Contains(suffix, "exit_code") || strings.Contains(suffix, "signal") {
		t.Errorf("diagnosticsSuffix with nil error should not contain exit_code/signal: %s", suffix)
	}
}

func TestDiagnosticsSuffix_NoLastEvent(t *testing.T) {
	suffix := diagnosticsSuffix(nil, parseResult{}, 0, "")
	if strings.Contains(suffix, "last_event=") {
		t.Errorf("empty parseResult should not include last_event: %s", suffix)
	}
	if strings.Contains(suffix, "last_event_excerpt=") {
		t.Errorf("empty parseResult should not include last_event_excerpt: %s", suffix)
	}
	if strings.Contains(suffix, "stderr_log=") {
		t.Errorf("empty stderr path should not include stderr_log: %s", suffix)
	}
}

func TestExcerpt_LastEventIntegration(t *testing.T) {
	// Integration: parseStream should set lastEventType and lastEventExcerpt
	// on every event in the stream.
	stream := strings.NewReader(strings.Join([]string{
		`{"type":"assistant","message":{"content":[{"type":"text","text":"Hello world"}]}}`,
		`{"type":"result","result":"Hello world","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
	}, "\n"))

	sink := &fake.SliceEventSink{}
	presult, err := parseStream(stream, sink)
	if err != nil {
		t.Fatalf("parseStream: %v", err)
	}

	// Last event should be the result event.
	if presult.lastEventType != "result" {
		t.Errorf("lastEventType = %q; want %q", presult.lastEventType, "result")
	}
	if presult.lastEventExcerpt == "" {
		t.Error("lastEventExcerpt should not be empty after successful parse")
	}
	if !strings.Contains(presult.lastEventExcerpt, "result") {
		t.Errorf("lastEventExcerpt should contain 'result': %q", presult.lastEventExcerpt)
	}
}
