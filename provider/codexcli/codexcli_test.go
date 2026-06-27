package codexcli

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

func TestCapabilities(t *testing.T) {
	p := New()
	if p.Name() != bridle.ProviderCodexCLI {
		t.Fatalf("Name = %q, want %q", p.Name(), bridle.ProviderCodexCLI)
	}
	caps := p.Capabilities()
	if caps.Category != bridle.CategorySubprocessStream {
		t.Errorf("Category = %q, want subprocess-stream", caps.Category)
	}
	if caps.SupportsCustomTools {
		t.Error("SupportsCustomTools should be false")
	}
	if caps.SupportsBeforeToolCall {
		t.Error("SupportsBeforeToolCall should be false")
	}
	if !caps.SupportsAfterToolCall {
		t.Error("SupportsAfterToolCall should be true")
	}
	if !caps.SupportsMCP {
		t.Error("SupportsMCP should be true (MCP wired via --config mcp_servers.*)")
	}
}

func TestNewDefaults(t *testing.T) {
	p := New()
	// Codex aspects run inside the funnel's trust boundary, so codex's
	// internal sandbox is disabled by default (its workspace-write default
	// blocks network / git push).
	if p.Sandbox != "danger-full-access" {
		t.Errorf("Sandbox = %q, want danger-full-access", p.Sandbox)
	}
	if p.ApprovalPolicy != "never" {
		t.Errorf("ApprovalPolicy = %q, want never", p.ApprovalPolicy)
	}
	// And it surfaces in the built args.
	args := strings.Join(p.buildCLIArgs(bridle.ProviderRequest{}), "\x00")
	if !strings.Contains(args, "--sandbox\x00danger-full-access") {
		t.Errorf("args missing --sandbox danger-full-access: %q", args)
	}
}

func TestBuildCLIArgs_NewAndResume(t *testing.T) {
	p := New()
	p.Sandbox = "read-only"
	p.Profile = "worker"
	p.ExtraConfig = []string{`model_reasoning_effort="low"`}

	req := bridle.ProviderRequest{
		Model: "gpt-5-codex",
		Cwd:   "/work",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "older"},
			{Role: "assistant", Content: "context"},
			{Role: "user", Content: "current"},
		},
	}

	args := strings.Join(p.buildCLIArgs(req), "\x00")
	for _, want := range []string{
		"--profile\x00worker",
		"--sandbox\x00read-only",
		"--ask-for-approval\x00never",
		"--config\x00model_reasoning_effort=\"low\"",
		"--cd\x00/work",
		"exec\x00--json",
		"--skip-git-repo-check",
		"--model\x00gpt-5-codex",
		"current",
	} {
		if !strings.Contains(args, want) {
			t.Fatalf("args missing %q in %q", want, args)
		}
	}

	req.Session = bridle.SessionHandle{ID: "019e8c6d-4415-70b0-942f-de66b23c4f10"}
	args = strings.Join(p.buildCLIArgs(req), "\x00")
	if !strings.Contains(args, "exec\x00resume\x00019e8c6d-4415-70b0-942f-de66b23c4f10\x00--json") {
		t.Fatalf("resume args not shaped correctly: %q", args)
	}

	req.Model = "default"
	args = strings.Join(p.buildCLIArgs(req), "\x00")
	if strings.Contains(args, "--model") {
		t.Fatalf("default model should omit --model: %q", args)
	}
}

func TestParseStream_TextOnly(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"thread.started","thread_id":"019e8c6d-4415-70b0-942f-de66b23c4f10"}`,
		`{"type":"turn.started"}`,
		`{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"hello"}}`,
		`{"type":"turn.completed","usage":{"input_tokens":22238,"cached_input_tokens":20352,"output_tokens":5,"reasoning_output_tokens":0}}`,
	}, "\n")

	sink := &fake.SliceEventSink{}
	result, err := parseStream(strings.NewReader(input), sink)
	if err != nil {
		t.Fatalf("parseStream error: %v", err)
	}
	if result.FinalText != "hello" {
		t.Fatalf("FinalText = %q, want hello", result.FinalText)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Fatalf("StopReason = %q, want model_done", result.StopReason)
	}
	if result.Usage.InputTokens != 22238 || result.Usage.CacheReadInputTokens != 20352 || result.Usage.OutputTokens != 5 {
		t.Fatalf("Usage = %+v", result.Usage)
	}
	if len(result.SessionDelta) != 2 {
		t.Fatalf("SessionDelta len = %d, want 2", len(result.SessionDelta))
	}
	norm, err := bridle.ParseSessionEvent(result.SessionDelta[0])
	if err != nil {
		t.Fatalf("ParseSessionEvent error: %v", err)
	}
	if !strings.Contains(norm.Content, "019e8c6d") {
		t.Fatalf("normalized init = %q", norm.Content)
	}
}

func TestParseStream_CommandExecution(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"thread.started","thread_id":"thread-1"}`,
		`{"type":"turn.started"}`,
		`{"type":"item.started","item":{"id":"item_0","type":"command_execution","command":"/bin/zsh -lc pwd","aggregated_output":"","exit_code":null,"status":"in_progress"}}`,
		`{"type":"item.completed","item":{"id":"item_0","type":"command_execution","command":"/bin/zsh -lc pwd","aggregated_output":"/work\n","exit_code":0,"status":"completed"}}`,
		`{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"done"}}`,
		`{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":4}}`,
	}, "\n")

	sink := &fake.SliceEventSink{}
	result, err := parseStream(strings.NewReader(input), sink)
	if err != nil {
		t.Fatalf("parseStream error: %v", err)
	}
	if result.StepCount != 1 {
		t.Fatalf("StepCount = %d, want 1", result.StepCount)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(result.ToolCalls))
	}
	call := result.ToolCalls[0]
	if call.ID != "item_0" || call.Name != "command_execution" {
		t.Fatalf("ToolCall = %+v", call)
	}
	var args map[string]string
	if err := json.Unmarshal(call.Args, &args); err != nil {
		t.Fatalf("call args unmarshal: %v", err)
	}
	if args["command"] != "/bin/zsh -lc pwd" {
		t.Fatalf("command arg = %q", args["command"])
	}

	var sawStart, sawResult, sawBoundary bool
	for _, ev := range sink.Events {
		switch ev.(type) {
		case bridle.ToolCallStart:
			sawStart = true
		case bridle.ToolCallResult:
			sawResult = true
		case bridle.StepBoundary:
			sawBoundary = true
		}
	}
	if !sawStart || !sawResult || !sawBoundary {
		t.Fatalf("events start=%v result=%v boundary=%v", sawStart, sawResult, sawBoundary)
	}
}

func TestRoundTripLive(t *testing.T) {
	if os.Getenv("BRIDLE_LIVE_CODEX") != "1" {
		t.Skip("set BRIDLE_LIVE_CODEX=1 to run live Codex CLI test")
	}
	if _, err := exec.LookPath("codex"); err != nil {
		t.Skip("codex CLI not on PATH")
	}

	p := New()
	p.Ephemeral = true
	p.Sandbox = "read-only"
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "default",
		UserMessage: "Reply with exactly the word: PONG",
		MaxSteps:    1,
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("RunTurn error: %v", err)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Fatalf("StopReason = %q, want model_done", result.StopReason)
	}
	if !strings.Contains(result.FinalText, "PONG") {
		t.Fatalf("FinalText = %q, want PONG", result.FinalText)
	}
}

// TestClassifyProviderError_Codex401IsAuth is the NEX-588 headline case:
// codex exits 1 on a 401 (expired auth.json / wrong endpoint) and prints
// only a bare HTTP code — no friendly "not logged in" string. It must
// classify as AUTH (so a credential-monitor can key off it, NEX-570),
// NOT the opaque subprocess-exit that forced manual in-pod debugging.
func TestClassifyProviderError_Codex401IsAuth(t *testing.T) {
	waitErr := &exec.ExitError{}
	cases := []struct {
		name   string
		stderr string
	}{
		{"bare 401", "ERROR: unexpected status 401 Unauthorized from https://api.openai.com"},
		{"401 in json", `{"error":{"code":401,"message":"Incorrect API key provided"}}`},
		{"unauthorized word", "request failed: Unauthorized"},
		{"expired token", "auth token expired; re-run codex login"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pe := classifyProviderError(tc.stderr, waitErr)
			if pe.Kind != bridle.ProviderErrorAuthFailed {
				t.Fatalf("Kind = %q, want %q (codex 401 must be AUTH, not generic exit)", pe.Kind, bridle.ProviderErrorAuthFailed)
			}
			if pe.Kind == bridle.ProviderErrorSubprocessExit {
				t.Fatal("regressed to opaque subprocess_exit")
			}
			if !strings.Contains(strings.ToLower(pe.Message), "auth") {
				t.Errorf("message not actionable for auth: %q", pe.Message)
			}
			t.Logf("kind=%s msg=%s", pe.Kind, pe.Message)
		})
	}
}

func TestClassifyProviderError_CodexActionableClasses(t *testing.T) {
	waitErr := &exec.ExitError{}
	cases := []struct {
		name     string
		stderr   string
		wantKind bridle.ProviderErrorKind
	}{
		{"cli-worded auth still wins", "Error: not logged in", bridle.ProviderErrorAuthFailed},
		{"rate limit 429", "HTTP 429: rate_limit_exceeded", bridle.ProviderErrorRateLimit},
		{"network", "dial tcp: connection refused", bridle.ProviderErrorNetworkError},
		{"missing binary", `exec: "codex": executable file not found in $PATH`, bridle.ProviderErrorConfig},
		{"crash", "Segmentation fault (core dumped)", bridle.ProviderErrorCrash},
		{"unmatched -> generic", "totally unrecognized blah", bridle.ProviderErrorSubprocessExit},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pe := classifyProviderError(tc.stderr, waitErr)
			if pe.Kind != tc.wantKind {
				t.Errorf("Kind = %q, want %q", pe.Kind, tc.wantKind)
			}
		})
	}
}

// writeFakeCodex writes an executable shell script that mimics the codex
// CLI for resume-robustness tests. When invoked with the `resume`
// subcommand it prints resumeStderr to stderr and exits 1 (simulating a
// missing/corrupt session); otherwise it emits a valid stream-json turn
// and exits 0. It records every invocation's argv into argvLog.
func writeFakeCodex(t *testing.T, resumeStderr, argvLog string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake-CLI shell script unsupported on windows")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "codex")
	script := `#!/bin/sh
echo "$@" >> "` + argvLog + `"
for a in "$@"; do
  if [ "$a" = "resume" ]; then
    echo '` + resumeStderr + `' 1>&2
    exit 1
  fi
done
echo '{"type":"thread.started","thread_id":"fresh-1"}'
echo '{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"hello from fresh"}}'
echo '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1,"reasoning_output_tokens":0}}'
exit 0
`
	if err := os.WriteFile(path, []byte(script), 0700); err != nil {
		t.Fatalf("write fake codex: %v", err)
	}
	return path
}

// TestRunTurn_ResumeNotFound_FallsBackToFresh is the NEX-588 resume
// robustness headline: a continuing turn resumes a session codex can't
// find; the provider must degrade to a fresh session (re-run without
// `resume`), emit a TurnErrorStageResumeFallback warning, and still
// return the fresh turn's content rather than hard-failing.
func TestRunTurn_ResumeNotFound_FallsBackToFresh(t *testing.T) {
	argvLog := filepath.Join(t.TempDir(), "argv.log")
	p := New()
	p.CodexPath = writeFakeCodex(t, "Error: no such thread missing-id", argvLog)

	sink := &fake.SliceEventSink{}
	result, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Session:  bridle.SessionHandle{ID: "missing-id", New: false},
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn returned error, want graceful fallback: %v", err)
	}
	if result.FinalText != "hello from fresh" {
		t.Fatalf("FinalText = %q, want fresh-session content", result.FinalText)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Errorf("StopReason = %q, want model_done", result.StopReason)
	}
	// A resume-fallback warning must have been surfaced.
	var sawFallback bool
	for _, ev := range sink.Events {
		if te, ok := ev.(bridle.TurnError); ok && te.Stage == bridle.TurnErrorStageResumeFallback {
			sawFallback = true
		}
	}
	if !sawFallback {
		t.Error("no TurnErrorStageResumeFallback warning emitted")
	}
	// The fake ran twice: once WITH resume (failed), once WITHOUT (fresh).
	log, _ := os.ReadFile(argvLog)
	invocations := strings.Count(string(log), "\n")
	if invocations != 2 {
		t.Errorf("fake invoked %d times, want 2 (resume then fresh)", invocations)
	}
	if strings.Count(string(log), "resume") != 1 {
		t.Errorf("expected exactly one resume invocation, log:\n%s", log)
	}
}

// TestRunTurn_ResumeTransientError_DoesNotFallBack ensures a transient/
// auth resume failure (NOT a missing session) is NOT silently degraded
// to a fresh session — it propagates so the caller can retry against the
// SAME session without dropping a builder's context.
func TestRunTurn_ResumeTransientError_DoesNotFallBack(t *testing.T) {
	argvLog := filepath.Join(t.TempDir(), "argv.log")
	p := New()
	// 401 → AUTH, not resume-not-found.
	p.CodexPath = writeFakeCodex(t, "Error: status 401 unauthorized", argvLog)

	sink := &fake.SliceEventSink{}
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Session:  bridle.SessionHandle{ID: "live-session", New: false},
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, sink)
	if err == nil {
		t.Fatal("expected error to propagate, got nil (must not silently drop session)")
	}
	if !bridle.IsProviderErrorKind(err, bridle.ProviderErrorAuthFailed) {
		t.Errorf("err kind not AUTH: %v", err)
	}
	// The fake ran exactly once — no fresh fallback.
	log, _ := os.ReadFile(argvLog)
	if n := strings.Count(string(log), "\n"); n != 1 {
		t.Errorf("fake invoked %d times, want 1 (no fallback on transient)", n)
	}
}
