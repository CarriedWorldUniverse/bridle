package claudecode_test

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
	"github.com/CarriedWorldUniverse/bridle/provider/claudecode"
)

// writeFakeClaude writes an executable shell script mimicking the claude
// CLI for resume-robustness tests (NEX-588). When invoked with --resume
// it prints resumeStderr to stderr and exits 1 (simulating a missing/
// corrupt session); otherwise it emits a valid stream-json turn and
// exits 0. Every invocation's argv is appended to argvLog.
func writeFakeClaude(t *testing.T, resumeStderr, argvLog string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake-CLI shell script unsupported on windows")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "claude")
	script := `#!/bin/sh
echo "$@" >> "` + argvLog + `"
for a in "$@"; do
  if [ "$a" = "--resume" ]; then
    echo '` + resumeStderr + `' 1>&2
    exit 1
  fi
done
echo '{"type":"system","subtype":"init","session_id":"fresh-1"}'
echo '{"type":"result","result":"hello from fresh","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}'
exit 0
`
	if err := os.WriteFile(path, []byte(script), 0700); err != nil {
		t.Fatalf("write fake claude: %v", err)
	}
	return path
}

// TestRunTurn_ResumeNotFound_FallsBackToFresh: a continuing turn resumes
// a session claude-code can't find; the provider degrades to a fresh
// session, emits a TurnErrorStageResumeFallback warning, and returns the
// fresh turn's content rather than hard-failing.
func TestRunTurn_ResumeNotFound_FallsBackToFresh(t *testing.T) {
	argvLog := filepath.Join(t.TempDir(), "argv.log")
	p := &claudecode.Provider{
		ClaudePath: writeFakeClaude(t, "Error: No conversation found with session id missing-id", argvLog),
	}

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
	if strings.Count(string(log), "--resume") != 1 {
		t.Errorf("expected exactly one --resume invocation:\n%s", log)
	}
}

// TestRunTurn_ResumeTransientError_DoesNotFallBack: an auth resume
// failure (NOT a missing session) propagates instead of silently
// degrading to fresh — the caller retries against the same session
// without dropping a builder's context.
func TestRunTurn_ResumeTransientError_DoesNotFallBack(t *testing.T) {
	argvLog := filepath.Join(t.TempDir(), "argv.log")
	p := &claudecode.Provider{
		ClaudePath: writeFakeClaude(t, "Error: status 401 unauthorized", argvLog),
	}

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
	log, _ := os.ReadFile(argvLog)
	if n := strings.Count(string(log), "\n"); n != 1 {
		t.Errorf("fake invoked %d times, want 1 (no fallback on transient):\n%s", n, log)
	}
}
