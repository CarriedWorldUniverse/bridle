package geminicli

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

func TestCapabilities(t *testing.T) {
	p := New()
	if p.Name() != bridle.ProviderGeminiCLI {
		t.Fatalf("Name = %q, want %q", p.Name(), bridle.ProviderGeminiCLI)
	}
	caps := p.Capabilities()
	if caps.Category != bridle.CategorySubprocessStream {
		t.Errorf("Category = %q, want subprocess-stream", caps.Category)
	}
}

func TestClassifyProviderError_ActionableClasses(t *testing.T) {
	waitErr := errors.New("exit status 1")
	cases := []struct {
		name     string
		stderr   string
		wantKind bridle.ProviderErrorKind
	}{
		{"401 auth", "Error: 401 Unauthorized", bridle.ProviderErrorAuthFailed},
		{"429 rate", "429 quota exceeded", bridle.ProviderErrorRateLimit},
		{"network", "dial tcp: connection refused", bridle.ProviderErrorNetworkError},
		{"missing binary", `exec: "gemini": executable file not found in $PATH`, bridle.ProviderErrorConfig},
		{"crash", "Segmentation fault (core dumped)", bridle.ProviderErrorCrash},
		{"unmatched -> generic", "weird unrecognized output", bridle.ProviderErrorSubprocessExit},
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

// writeFakeGemini writes an executable shell script mimicking the gemini
// CLI for resume-robustness tests (NEX-588). When invoked with --resume
// it prints resumeStderr to stderr and exits 1; otherwise it emits a
// valid stream-json turn and exits 0. argv is logged to argvLog.
func writeFakeGemini(t *testing.T, resumeStderr, argvLog string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake-CLI shell script unsupported on windows")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "gemini")
	script := `#!/bin/sh
echo "$@" >> "` + argvLog + `"
for a in "$@"; do
  if [ "$a" = "--resume" ]; then
    echo '` + resumeStderr + `' 1>&2
    exit 1
  fi
done
echo '{"type":"init","session_id":"fresh-1","model":"gemini"}'
echo '{"type":"message","role":"assistant","content":"hello from fresh"}'
echo '{"type":"result","status":"success","stats":{"input_tokens":1,"output_tokens":1}}'
exit 0
`
	if err := os.WriteFile(path, []byte(script), 0700); err != nil {
		t.Fatalf("write fake gemini: %v", err)
	}
	return path
}

func TestRunTurn_ResumeNotFound_FallsBackToFresh(t *testing.T) {
	argvLog := filepath.Join(t.TempDir(), "argv.log")
	p := New()
	p.GeminiPath = writeFakeGemini(t, "invalid resume index: 99", argvLog)

	sink := &fake.SliceEventSink{}
	result, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Session:  bridle.SessionHandle{ID: "99"},
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn returned error, want graceful fallback: %v", err)
	}
	if result.FinalText != "hello from fresh" {
		t.Fatalf("FinalText = %q, want fresh-session content", result.FinalText)
	}

	var sawFallback bool
	for _, ev := range sink.Events {
		if te, ok := ev.(bridle.TurnError); ok && te.Stage == bridle.TurnErrorStageResumeFallback {
			sawFallback = true
		}
	}
	if !sawFallback {
		t.Error("no TurnErrorStageResumeFallback warning emitted")
	}

	log, _ := os.ReadFile(argvLog)
	if n := strings.Count(string(log), "\n"); n != 2 {
		t.Errorf("fake invoked %d times, want 2 (resume then fresh):\n%s", n, log)
	}
}

func TestRunTurn_ResumeTransientError_DoesNotFallBack(t *testing.T) {
	argvLog := filepath.Join(t.TempDir(), "argv.log")
	p := New()
	p.GeminiPath = writeFakeGemini(t, "Error: 401 unauthorized", argvLog)

	sink := &fake.SliceEventSink{}
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Session:  bridle.SessionHandle{ID: "latest"},
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, sink)
	if err == nil {
		t.Fatal("expected error to propagate, got nil")
	}
	if !bridle.IsProviderErrorKind(err, bridle.ProviderErrorAuthFailed) {
		t.Errorf("err kind not AUTH: %v", err)
	}
	log, _ := os.ReadFile(argvLog)
	if n := strings.Count(string(log), "\n"); n != 1 {
		t.Errorf("fake invoked %d times, want 1 (no fallback on transient):\n%s", n, log)
	}
}
