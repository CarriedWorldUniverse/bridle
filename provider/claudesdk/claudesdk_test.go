// Deterministic tests against a FAKE sidecar (a POSIX shell script
// speaking the wire protocol — no real @anthropic-ai/claude-agent-sdk,
// no subscription, no network). Precedent: provider/claudecode's
// writeFakeClaude (claudecode_resume_test.go) and
// fake/subprocess_provider.go. See bridle-agentsdk-spec.md §10 for the
// acceptance criteria these pin.
package claudesdk_test

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
	"github.com/CarriedWorldUniverse/bridle/provider/claudesdk"
)

// writeFakeSidecar writes an executable POSIX shell script that speaks
// the bridle<->sidecar wire protocol (reads one init line, then whatever
// the caller's body script does). Mirrors claudecode's writeFakeClaude.
func writeFakeSidecar(t *testing.T, body string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake-sidecar shell script unsupported on windows")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "bridle-claude-sidecar")
	script := "#!/bin/sh\nread init_line\n" + body + "\n"
	if err := os.WriteFile(path, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake sidecar: %v", err)
	}
	return path
}

// --- §10.1 (shape): funnel-mode custom tool round trip through the
// REAL harness — the same ProviderRequest.ToolExecutor wiring run.go
// installs for any live caller (NEX-745 §4), not a hand-built stand-in.

func TestClaudeSDK_FunnelToolRoundTrip(t *testing.T) {
	sidecar := writeFakeSidecar(t, `
echo '{"type":"tool_call","id":"call-1","name":"echo","args":{"msg":"hi"}}'
read tool_result_line
echo '{"type":"text_delta","text":"echo tool returned: pong"}'
echo '{"type":"usage","input_tokens":5,"output_tokens":3}'
echo '{"type":"done","stop_reason":"end_turn","session_id":"sess-1","model":"claude-fake"}'
`)

	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"echo": {{Result: json.RawMessage(`"pong"`)}},
	})

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "claude-fake",
		UserMessage: "call echo",
		Tools: []bridle.ToolDef{{
			Name:        "echo",
			Description: "echoes back",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"msg":{"type":"string"}}}`),
		}},
	}, runner, sink)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if result.FinalText != "echo tool returned: pong" {
		t.Errorf("FinalText = %q; want the post-tool text (final answer depends on the tool result)", result.FinalText)
	}

	// The tool_call/tool_result pair must be visible in bridle's event
	// log (spec §10.1) — emitted by the REAL executeToolCall path via
	// ToolExecutor, proving the harness (not the provider) executed it.
	var sawStart, sawResult bool
	for _, ev := range sink.Events {
		if s, ok := ev.(bridle.ToolCallStart); ok && s.ID == "call-1" && s.Name == "echo" {
			sawStart = true
		}
		if r, ok := ev.(bridle.ToolCallResult); ok && r.ID == "call-1" && string(r.Result) == `"pong"` {
			sawResult = true
		}
	}
	if !sawStart {
		t.Error("no ToolCallStart{ID:call-1,Name:echo} event seen")
	}
	if !sawResult {
		t.Error("no ToolCallResult{ID:call-1,Result:\"pong\"} event seen")
	}

	// By design (see run.go's NEX-745 comment + claudesdk package doc):
	// the tool round-trip already completed INSIDE this one RunTurn call
	// via ToolExecutor, so ProviderResult/TurnResult.ToolCalls comes back
	// EMPTY — returning it populated would make run.go's per-round tool
	// loop re-execute the same call a second time.
	if len(result.ToolCalls) != 0 {
		t.Errorf("TurnResult.ToolCalls len = %d; want 0 (already executed via ToolExecutor, not re-surfaced for run.go to re-run)", len(result.ToolCalls))
	}
	if result.Usage.InputTokens != 5 || result.Usage.OutputTokens != 3 {
		t.Errorf("Usage = %+v; want {5,3} from the sidecar's usage event", result.Usage)
	}
}

// --- §10.6: refusal maps to error class "refusal", non-retryable.

func TestClaudeSDK_RefusalMapsToRefusalClass(t *testing.T) {
	sidecar := writeFakeSidecar(t, `
echo '{"type":"done","stop_reason":"refusal","session_id":"sess-1"}'
`)
	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "claude-fake",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "do something unsafe"}},
	}, sink)
	if err == nil {
		t.Fatal("expected an error for a refusal stop_reason, got nil")
	}
	if !bridle.IsProviderErrorKind(err, bridle.ProviderErrorRefusal) {
		t.Errorf("err = %v; want ProviderErrorKind refusal", err)
	}
}

func TestClaudeSDK_RefusalNotRetried(t *testing.T) {
	argvLog := filepath.Join(t.TempDir(), "argv.log")
	sidecar := writeFakeSidecar(t, `
echo "invoked" >> "`+argvLog+`"
echo '{"type":"done","stop_reason":"refusal"}'
`)
	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel, MaxRetries: 3}
	sink := &fake.SliceEventSink{}

	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "claude-fake",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, sink)
	if err == nil {
		t.Fatal("expected refusal error")
	}
	log, _ := os.ReadFile(argvLog)
	if got := len(splitNonEmptyLines(string(log))); got != 1 {
		t.Errorf("sidecar invoked %d times; want 1 (refusal must not be retried even with MaxRetries>0)", got)
	}
}

// --- MED (NEX-745 review gate): ambient ANTHROPIC_API_KEY/
// ANTHROPIC_AUTH_TOKEN must NOT reach the sidecar's env. The vendored
// Agent SDK's auth precedence ranks ANTHROPIC_API_KEY ABOVE
// CLAUDE_CODE_OAUTH_TOKEN, so a dual-lane deployment (claude-api lane in
// the same bridle process reading ANTHROPIC_API_KEY from env) would
// otherwise silently fall back to METERED api-key billing for every
// claudesdk turn too, defeating the whole point of the subscription
// lane (spec §10.2).

func TestClaudeSDK_SidecarEnv_ScrubsAmbientAnthropicAuthVars(t *testing.T) {
	envFile := filepath.Join(t.TempDir(), "env.out")
	sidecar := writeFakeSidecar(t, `
env > "`+envFile+`"
echo '{"type":"done","stop_reason":"end_turn"}'
`)

	t.Setenv("ANTHROPIC_API_KEY", "sk-ambient-metered-key-must-not-leak")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "ambient-auth-token-must-not-leak")

	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:       "claude-fake",
		Messages:    []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		ProviderEnv: map[string]string{"CLAUDE_CODE_OAUTH_TOKEN": "sub-token-authoritative"},
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	envOut, readErr := os.ReadFile(envFile)
	if readErr != nil {
		t.Fatalf("read captured sidecar env: %v", readErr)
	}
	env := string(envOut)
	if strings.Contains(env, "ANTHROPIC_API_KEY=") {
		t.Error("sidecar env contains ANTHROPIC_API_KEY — the ambient metered key must be scrubbed, not silently override CLAUDE_CODE_OAUTH_TOKEN")
	}
	if strings.Contains(env, "ANTHROPIC_AUTH_TOKEN=") {
		t.Error("sidecar env contains ANTHROPIC_AUTH_TOKEN — must be scrubbed")
	}
	if !strings.Contains(env, "CLAUDE_CODE_OAUTH_TOKEN=sub-token-authoritative") {
		t.Error("sidecar env missing CLAUDE_CODE_OAUTH_TOKEN — the subscription token from ProviderEnv must still reach the sidecar")
	}
}

// --- §10.4: SIGTERM mid-turn -> partial ProviderResult, no orphan process.

func TestClaudeSDK_KillMidTurn_NoOrphan(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("pgrep-based orphan check is unix-only")
	}
	sidecar := writeFakeSidecar(t, `
echo '{"type":"text_delta","text":"partial"}'
exec sleep 9999
`)
	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel shortly after the sidecar has had time to emit its one
	// event and settle into the sleep. Not perfectly deterministic on
	// timing, but generous enough that it isn't flaky, and the ASSERTION
	// (no lingering process matching this test's unique sidecar path) is
	// what actually proves the kill/no-orphan invariant, independent of
	// exactly when cancel() fires relative to the sleep starting.
	go func() {
		time.Sleep(200 * time.Millisecond)
		cancel()
	}()

	result, err := p.RunTurn(ctx, bridle.ProviderRequest{
		Model:    "claude-fake",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn on cancellation: want nil error (clean interrupt), got %v", err)
	}
	if result.StopReason != bridle.StopReasonAborted {
		t.Errorf("StopReason = %q; want aborted", result.StopReason)
	}
	if result.FinalText != "partial" {
		t.Errorf("FinalText = %q; want the partial text emitted before cancellation", result.FinalText)
	}

	// No orphaned process left behind: pgrep for anything whose command
	// line references this test's unique fake-sidecar path (t.TempDir()
	// is unique per test run). Bracket-trick the search pattern so pgrep
	// doesn't match its own invocation's argv.
	dir := filepath.Dir(sidecar)
	pattern := "[" + dir[:1] + "]" + dir[1:] // bracket-trick the leading char so pgrep's own argv (which contains dir unbracketed) doesn't self-match
	waitForNoOrphan(t, pattern)
}

// --- HIGH (NEX-745 review gate): SIGTERM must reap the whole process
// TREE, not just the sidecar's own PID. The real bridle-claude-sidecar
// spawns the actual `claude` CLI as a GRANDCHILD (the Agent SDK's own
// subprocess), and index.ts registers no SIGTERM handler of its own, so
// a single-PID kill leaves that grandchild running — orphaned, still
// holding the auth token, possibly still streaming/spending. The
// existing TestClaudeSDK_KillMidTurn_NoOrphan test above uses a 1-HOP
// fake (the fake sidecar script IS the leaf process) so it can't catch
// this: this test's fake sidecar spawns its OWN child (a background
// `tail -f` on a marker file whose path is unique per test run,
// pgrep-able independent of the sidecar's own pid/argv) to reproduce the
// real 2-hop topology.

func TestClaudeSDK_KillMidTurn_NoOrphanGrandchild(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("pgrep-based orphan check is unix-only")
	}
	sidecarDir := t.TempDir()
	marker := filepath.Join(sidecarDir, "grandchild-marker")
	if err := os.WriteFile(marker, []byte("x"), 0o600); err != nil {
		t.Fatalf("write marker: %v", err)
	}
	sidecarPath := filepath.Join(sidecarDir, "bridle-claude-sidecar")
	// The sidecar backgrounds a grandchild (`tail -f marker`, a
	// long-running process whose argv carries the unique marker path)
	// and then waits on it — the same shape as node spawning the real
	// `claude` CLI and awaiting its exit.
	// The grandchild's stdout/stderr are redirected away from the
	// sidecar's own stdout pipe: on a real fork, a backgrounded child
	// inherits its parent's fds, and if it kept the pipe's write end
	// open, bridle's stdout-EOF read loop would block on the grandchild
	// too (a real but SEPARATE bug from the one under test here — the
	// real claude CLI is spawned via node's child_process with its own
	// stdio, not sharing the sidecar's stdout). Redirecting isolates
	// this test to the kill/reap behavior alone.
	script := "#!/bin/sh\nread init_line\n" +
		`echo '{"type":"text_delta","text":"partial"}'` + "\n" +
		`tail -f "` + marker + `" >/dev/null 2>&1 &` + "\n" +
		`wait` + "\n"
	if err := os.WriteFile(sidecarPath, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake sidecar: %v", err)
	}

	p := &claudesdk.Provider{SidecarPath: sidecarPath, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(200 * time.Millisecond)
		cancel()
	}()

	// Bounded by an explicit timeout, not just the default `go test`
	// timeout: RunTurn's own stdout-EOF read loop is a separate risk
	// surface from the orphan bug this test targets, and a bounded
	// failure here is far more actionable than a 10-minute hang.
	type runResult struct {
		result bridle.ProviderResult
		err    error
	}
	runDone := make(chan runResult, 1)
	go func() {
		res, err := p.RunTurn(ctx, bridle.ProviderRequest{
			Model:    "claude-fake",
			Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		}, sink)
		runDone <- runResult{res, err}
	}()

	var result bridle.ProviderResult
	var err error
	select {
	case rr := <-runDone:
		result, err = rr.result, rr.err
	case <-time.After(15 * time.Second):
		t.Fatal("RunTurn did not return within 15s of cancellation — the sidecar's stdout pipe likely never hit EOF")
	}
	if err != nil {
		t.Fatalf("RunTurn on cancellation: want nil error (clean interrupt), got %v", err)
	}
	if result.StopReason != bridle.StopReasonAborted {
		t.Errorf("StopReason = %q; want aborted", result.StopReason)
	}

	// Assert BOTH hops are gone: the sidecar script itself, AND its
	// grandchild (the tail process pinned to the marker path). A
	// single-PID kill (the pre-fix behavior) leaves the second alive.
	sidecarPattern := "[" + filepath.Dir(sidecarPath)[:1] + "]" + filepath.Dir(sidecarPath)[1:]
	waitForNoOrphan(t, sidecarPattern)
	markerPattern := "[" + marker[:1] + "]" + marker[1:]
	waitForNoOrphan(t, markerPattern)
}

// waitForNoOrphan polls pgrep -f <pattern> for up to a few seconds
// (covers WatchCancel's grace window if SIGTERM alone didn't land) and
// fails if a matching process is still alive at the end.
func waitForNoOrphan(t *testing.T, pattern string) {
	t.Helper()
	deadline := time.Now().Add(8 * time.Second)
	for {
		out, err := exec.Command("pgrep", "-f", pattern).Output()
		if err != nil {
			return // pgrep exits non-zero when nothing matches — no orphan
		}
		if time.Now().After(deadline) {
			t.Fatalf("orphaned process still matches %q after grace period:\n%s", pattern, out)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func splitNonEmptyLines(s string) []string {
	var out []string
	start := 0
	for i, c := range s {
		if c == '\n' {
			if i > start {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}
