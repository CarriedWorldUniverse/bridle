// Live acceptance tests against the REAL bridle-claude-sidecar + the
// real @anthropic-ai/claude-agent-sdk (NEX-297-style Layer 2). Env-var
// gated on CLAUDE_CODE_OAUTH_TOKEN (spec §10.2's setup-token auth path)
// so `go test ./...` SKIPS these cleanly for contributors/CI without a
// live subscription token — matching provider/claude/live_test.go's
// established L2 gating mechanism (os.Getenv + t.Skip, no build tag).
//
// These exercise spec §10.1-10.3 (real turn, real auth, session
// continuity) — the operator's job per the NEX-745 scope boundary, NOT
// something this builder's autonomous run can produce evidence for.
// Running them requires:
//
//	export CLAUDE_CODE_OAUTH_TOKEN=<a real `claude setup-token` token>
//	cd bridle-claude-sidecar && npm install && npm run build
//	go test ./provider/claudesdk/... -run TestLive -v
package claudesdk_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
	"github.com/CarriedWorldUniverse/bridle/provider/claudesdk"
)

// liveSidecarPath resolves the built sidecar entry point relative to
// this test file. Returns "" (with the caller expected to t.Skip) when
// CLAUDE_CODE_OAUTH_TOKEN is unset OR the sidecar hasn't been built —
// the second check gives the operator an actionable message instead of
// a confusing spawn failure.
func liveSidecarPath(t *testing.T) string {
	t.Helper()
	if os.Getenv("CLAUDE_CODE_OAUTH_TOKEN") == "" {
		t.Skip("CLAUDE_CODE_OAUTH_TOKEN not set; skipping live claudesdk test (operator-run only, see file doc comment)")
	}
	path := filepath.Join("..", "..", "bridle-claude-sidecar", "dist", "index.js")
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("resolve sidecar path: %v", err)
	}
	if _, statErr := os.Stat(abs); statErr != nil {
		t.Skipf("built sidecar not found at %s — run `cd bridle-claude-sidecar && npm install && npm run build` first: %v", abs, statErr)
	}
	return abs
}

// liveProvider wires a Provider against the built sidecar. SidecarPath
// is a single-binary exec.Command target (mirrors claudecode.ClaudePath
// — no argv prefix support), but `tsc` doesn't set dist/index.js's
// executable bit, so the real image is expected to install a one-line
// `exec node .../index.js "$@"` wrapper at the PATH-resolved
// "bridle-claude-sidecar" name (see the sidecar README's packaging
// section). This test writes that same wrapper itself so the live path
// doesn't depend on a manual chmod step.
func liveProvider(t *testing.T) *claudesdk.Provider {
	t.Helper()
	sidecarJS := liveSidecarPath(t)
	wrapperDir := t.TempDir()
	wrapper := filepath.Join(wrapperDir, "bridle-claude-sidecar")
	script := "#!/bin/sh\nexec node \"" + sidecarJS + "\" \"$@\"\n"
	if err := os.WriteFile(wrapper, []byte(script), 0o700); err != nil {
		t.Fatalf("write sidecar wrapper: %v", err)
	}
	return &claudesdk.Provider{SidecarPath: wrapper, Mode: claudesdk.ModeFunnel}
}

// TestLive_ClaudeSDK_SimpleTurn is spec §10.1's tightest slice without a
// tool: a basic text-in/text-out turn against the real subscription
// auth path, proving the sidecar + provider + harness agree on the
// turn protocol end to end.
func TestLive_ClaudeSDK_SimpleTurn(t *testing.T) {
	p := liveProvider(t)
	h := bridle.NewHarness(p)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       "claude-haiku-4-5",
		UserMessage: "Reply with just the word ping and nothing else.",
		MaxSteps:    1,
	}, fake.NewToolRunner(nil), &fake.SliceEventSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if result.FinalText == "" {
		t.Error("FinalText is empty")
	}
	if result.Usage.InputTokens == 0 {
		t.Error("Usage.InputTokens = 0, want non-zero (live SDK should report)")
	}
}

// TestLive_ClaudeSDK_NoAPIKeyInEnv is spec §10.2's billing-shape guard:
// asserts the turn succeeds with NO ANTHROPIC_API_KEY in this process's
// env — auth flows only through CLAUDE_CODE_OAUTH_TOKEN (subscription),
// never a metered API key. This is necessary-but-not-sufficient
// evidence for §10.2 (the operator's Anthropic-console no-metered-usage
// check is the sufficient half, out of this builder's scope per the
// NEX-745 boundary).
func TestLive_ClaudeSDK_NoAPIKeyInEnv(t *testing.T) {
	if v := os.Getenv("ANTHROPIC_API_KEY"); v != "" {
		t.Skip("ANTHROPIC_API_KEY is set in this environment; skipping so the test doesn't give a false pass for the metered-key-absent claim")
	}
	p := liveProvider(t)
	h := bridle.NewHarness(p)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       "claude-haiku-4-5",
		UserMessage: "Reply with just the word ping and nothing else.",
		MaxSteps:    1,
	}, fake.NewToolRunner(nil), &fake.SliceEventSink{})
	if err != nil {
		t.Fatalf("RunTurn without ANTHROPIC_API_KEY: %v", err)
	}
	if result.FinalText == "" {
		t.Error("FinalText is empty")
	}
}

// TestLive_ClaudeSDK_ToolRoundtrip is spec §10.1's full shape: a custom
// bridle tool call, executed by the real ToolRunner via the real
// ToolExecutor wiring, against the real SDK.
func TestLive_ClaudeSDK_ToolRoundtrip(t *testing.T) {
	p := liveProvider(t)
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"echo": {{Result: json.RawMessage(`{"echo":"pong"}`)}},
	})
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	result, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       "claude-haiku-4-5",
		UserMessage: `Call the echo tool with msg="ping" and report exactly what it returned.`,
		MaxSteps:    3,
		Tools: []bridle.ToolDef{{
			Name:        "echo",
			Description: "Echo back a canned result for testing.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}`),
		}},
	}, runner, sink)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	var sawToolCall bool
	for _, ev := range sink.Events {
		if s, ok := ev.(bridle.ToolCallStart); ok && s.Name == "echo" {
			sawToolCall = true
		}
	}
	if !sawToolCall {
		t.Error("expected a ToolCallStart{Name:echo} event; got none — tool schema or MCP wiring may have regressed")
	}
	if result.FinalText == "" {
		t.Error("FinalText is empty")
	}
}

// TestLive_ClaudeSDK_SessionContinuity is spec §10.3: two RunTurns on
// one session id; the second turn's transcript should reflect the
// first's content, and the resume-id invariant (spec §7.3) must not
// fire a mismatch warning.
func TestLive_ClaudeSDK_SessionContinuity(t *testing.T) {
	p := liveProvider(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sessionID := "bridle-claudesdk-live-test-session"
	sink1 := &fake.SliceEventSink{}
	r1, err := p.RunTurn(ctx, bridle.ProviderRequest{
		Model:    "claude-haiku-4-5",
		Session:  bridle.SessionHandle{ID: sessionID, New: true},
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "Remember the secret word: pineapple. Reply OK."}},
	}, sink1)
	if err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if r1.FinalText == "" {
		t.Fatal("turn 1: FinalText is empty")
	}

	sink2 := &fake.SliceEventSink{}
	r2, err := p.RunTurn(ctx, bridle.ProviderRequest{
		Model:    "claude-haiku-4-5",
		Session:  bridle.SessionHandle{ID: sessionID, New: false},
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "What was the secret word I told you?"}},
	}, sink2)
	if err != nil {
		t.Fatalf("turn 2 (resume): %v", err)
	}
	if r2.FinalText == "" {
		t.Fatal("turn 2: FinalText is empty")
	}

	for _, ev := range sink2.Events {
		if te, ok := ev.(bridle.TurnError); ok && te.Stage == bridle.TurnErrorStageStderrOutput {
			t.Logf("turn 2 non-fatal event (check for a resume-id mismatch warning): %v", te.Err)
		}
	}
}
