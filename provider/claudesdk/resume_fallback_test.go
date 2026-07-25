package claudesdk_test

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
	"github.com/CarriedWorldUniverse/bridle/provider/claudesdk"
)

// writeResumeAwareSidecar writes a fake sidecar that behaves DIFFERENTLY
// depending on whether the init line asks to resume a session: a resume
// attempt reports the CLI's real not-found error, a fresh start succeeds.
// That is exactly the shape of a stale/expired session id, and it needs a
// stateful-ish fake because the whole point is that attempt 2 must differ
// from attempt 1.
func writeResumeAwareSidecar(t *testing.T, notFoundMsg string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake-sidecar shell script unsupported on windows")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "bridle-claude-sidecar")
	script := `#!/bin/sh
read init_line
case "$init_line" in
  *'"resume"'*)
    printf '%s\n' '{"type":"error","class":"provider","message":"` + notFoundMsg + `"}'
    ;;
  *)
    printf '%s\n' '{"type":"text_delta","text":"FRESH_SESSION_ANSWER"}'
    printf '%s\n' '{"type":"usage","input_tokens":3,"output_tokens":2}'
    printf '%s\n' '{"type":"done","stop_reason":"end_turn","session_id":"sess-fresh","model":"claude-fake"}'
    ;;
esac
`
	if err := os.WriteFile(path, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake sidecar: %v", err)
	}
	return path
}

// Regression test for agora#120: a stale session id used to fail the
// turn outright, which permanently broke any thread that accumulated
// one — including agora's default thread.
func TestClaudeSDK_ResumeNotFound_FallsBackToAFreshSession(t *testing.T) {
	sidecar := writeResumeAwareSidecar(t,
		"Claude Code returned an error result: No conversation found with session ID: 0c020e62-f27a-5f1c-8e37-8cf9e413a326")
	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	result, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "claude-fake",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "continue please"}},
		Session:  bridle.SessionHandle{ID: "sess-STALE", New: false},
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn: %v — a missing session must degrade to a fresh one, not fail the turn", err)
	}
	if result.FinalText != "FRESH_SESSION_ANSWER" {
		t.Fatalf("FinalText = %q; want the fresh session's real answer", result.FinalText)
	}

	var sawFallback bool
	for _, ev := range sink.Events {
		te, ok := ev.(bridle.TurnError)
		if !ok || te.Stage != bridle.TurnErrorStageResumeFallback {
			continue
		}
		if strings.Contains(te.Err.Error(), "sess-STALE") {
			sawFallback = true
		}
	}
	if !sawFallback {
		t.Error("no TurnError{Stage:ResumeFallback} naming the stale id — the lost context must be VISIBLE, not silent")
	}
}

// The fallback must not fire for a NEW session (nothing to fall back
// from) — that would mask a genuine first-turn failure as a retry.
func TestClaudeSDK_NewSession_DoesNotTriggerResumeFallback(t *testing.T) {
	sidecar := writeFakeSidecar(t, `
echo '{"type":"error","class":"provider","message":"No conversation found with session ID: whatever"}'
`)
	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "claude-fake",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		Session:  bridle.SessionHandle{ID: "sess-new", New: true},
	}, sink)
	if err == nil {
		t.Fatal("a NEW session reporting not-found must still fail — there is no prior session to fall back from")
	}
	for _, ev := range sink.Events {
		if te, ok := ev.(bridle.TurnError); ok && te.Stage == bridle.TurnErrorStageResumeFallback {
			t.Error("resume-fallback fired for a New session")
		}
	}
}

// A transient/auth error on a resuming turn must NOT be mistaken for a
// missing session: retrying against the SAME session is what preserves a
// caller's context across a blip, and silently dropping it would lose
// real work.
func TestClaudeSDK_TransientErrorOnResume_DoesNotDropTheSession(t *testing.T) {
	sidecar := writeFakeSidecar(t, `
echo '{"type":"error","class":"auth","message":"authentication failed"}'
`)
	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "claude-fake",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "continue"}},
		Session:  bridle.SessionHandle{ID: "sess-live", New: false},
	}, sink)
	if err == nil {
		t.Fatal("an auth error should still surface")
	}
	for _, ev := range sink.Events {
		if te, ok := ev.(bridle.TurnError); ok && te.Stage == bridle.TurnErrorStageResumeFallback {
			t.Error("an auth failure was misread as a missing session and silently dropped the caller's context")
		}
	}
}

// If the FRESH attempt itself reports not-found, the guard must stop
// rather than loop forever.
func TestClaudeSDK_ResumeFallback_DoesNotLoop(t *testing.T) {
	// Always reports not-found, resume or not.
	sidecar := writeFakeSidecar(t, `
echo '{"type":"error","class":"provider","message":"No conversation found with session ID: x"}'
`)
	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
			Model:    "claude-fake",
			Messages: []bridle.ProviderMessage{{Role: "user", Content: "continue"}},
			Session:  bridle.SessionHandle{ID: "sess-stale", New: false},
		}, sink)
		if err == nil {
			t.Error("a persistently-missing session must eventually fail, not succeed")
		}
	}()

	// Bounded rather than a bare wait on done: an infinite loop would
	// otherwise hang the test, reading as a stuck CI job rather than a
	// bug report.
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("RunTurn did not return — the resume fallback is looping instead of giving up")
	}

	// Exactly ONE fallback attempt: the guard must not let it re-fire.
	var fallbacks int
	for _, ev := range sink.Events {
		if te, ok := ev.(bridle.TurnError); ok && te.Stage == bridle.TurnErrorStageResumeFallback {
			fallbacks++
		}
	}
	if fallbacks != 1 {
		t.Fatalf("resume fallback fired %d times; want exactly 1", fallbacks)
	}
}
