// Live integration tests against real Anthropic-shape endpoints
// (NEX-297 Layer 2). Env-var gated — each subtest skips cleanly when
// the relevant API key is absent, so `go test ./...` works for
// contributors without keys + fires the real checks in CI / local
// dev for those who have them.
//
// Covers what NEX-297 L1's manual smoke validated by hand:
//   - simple text turn against real api.anthropic.com
//   - simple text turn against DeepSeek /anthropic compat endpoint
//   - tool roundtrip on both (regression guard for NEX-299 Pass 1's
//     tool-schema fix — DeepSeek's strict validator rejected the
//     pre-fix shape; real Anthropic was lenient)
//   - streaming event shape (chunks arrive in order, terminal TurnDone)
//   - auth failure surfaces cleanly as a Go error
//   - model-not-found surfaces cleanly
//
// Each test iterates the shapeConfigs table; subtests skip
// independently so operators with only one key still get partial
// coverage.

package claude_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/claude"
)

// shapeConfig parameterises a live test across the Anthropic-API-
// compatible endpoints we care about. Each entry skips when its
// keyEnv is unset, so contributors / CI without that particular
// key still get the rest of the suite.
type shapeConfig struct {
	name    string // subtest name (anthropic, deepseek-anthropic, etc.)
	keyEnv  string // env var holding the API key
	baseURL string // "" = SDK default (api.anthropic.com)
	model   string // model id for simple text + tool tests
}

var shapeConfigs = []shapeConfig{
	{
		name:    "anthropic",
		keyEnv:  "ANTHROPIC_API_KEY",
		baseURL: "",
		model:   "claude-haiku-4-5",
	},
	{
		name:    "deepseek_anthropic",
		keyEnv:  "DEEPSEEK_ANTHROPIC_API_KEY",
		baseURL: "https://api.deepseek.com/anthropic",
		model:   "deepseek-chat",
	},
}

func liveProvider(t *testing.T, c shapeConfig) (*claude.Provider, context.Context, context.CancelFunc) {
	t.Helper()
	key := os.Getenv(c.keyEnv)
	if key == "" {
		t.Skipf("%s not set; skipping live test for shape=%s", c.keyEnv, c.name)
	}
	p := claude.NewWithBaseURL(key, c.baseURL)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	return p, ctx, cancel
}

// TestLive_Claude_SimpleTurn pins that a basic text-in / text-out
// turn works end-to-end against the real endpoint. Tightest possible
// validation that the provider, harness, and remote API agree on the
// turn protocol — token count + non-empty FinalText + clean stop.
func TestLive_Claude_SimpleTurn(t *testing.T) {
	for _, c := range shapeConfigs {
		t.Run(c.name, func(t *testing.T) {
			p, ctx, cancel := liveProvider(t, c)
			defer cancel()
			h := bridle.NewHarness(p)
			result, err := h.RunTurn(ctx, bridle.TurnRequest{
				Model:       c.model,
				UserMessage: "Reply with just the word ping and nothing else.",
				MaxSteps:    1,
			}, noopRunner{}, nullSink{})
			if err != nil {
				t.Fatalf("RunTurn: %v", err)
			}
			if strings.TrimSpace(result.FinalText) == "" {
				t.Errorf("FinalText is empty")
			}
			if result.Usage.InputTokens == 0 {
				t.Errorf("Usage.InputTokens = 0, want non-zero (live API should report)")
			}
			if result.StopReason == "" {
				t.Errorf("StopReason is empty, want non-empty")
			}
		})
	}
}

// TestLive_Claude_ToolRoundtrip is the NEX-299 Pass 1 regression
// guard: tool schemas must serialise into a shape strict validators
// accept. Pre-fix, DeepSeek's /anthropic rejected the malformed
// nested-properties payload with a 400; real Anthropic accepted it
// via lenient parsing. This test exercises the full roundtrip —
// model calls the echo tool, runner returns the args, model reports
// back — across both endpoints so a future regression in
// toClaudeTools would surface on at least one variant.
func TestLive_Claude_ToolRoundtrip(t *testing.T) {
	for _, c := range shapeConfigs {
		t.Run(c.name, func(t *testing.T) {
			p, ctx, cancel := liveProvider(t, c)
			defer cancel()
			h := bridle.NewHarness(p)
			result, err := h.RunTurn(ctx, bridle.TurnRequest{
				Model:       c.model,
				UserMessage: `Call the echo tool with text="ping" and report what you got back.`,
				MaxSteps:    3,
				Tools: []bridle.ToolDef{{
					Name:        "echo",
					Description: "Echo back the input as a tool result.",
					InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`),
				}},
			}, echoRunner{}, nullSink{})
			if err != nil {
				t.Fatalf("RunTurn: %v", err)
			}
			if len(result.ToolCalls) == 0 {
				t.Errorf("expected at least one tool call; got 0 — schema may have regressed (NEX-299 Pass 1)")
			}
			if result.ToolCalls[0].Name != "echo" {
				t.Errorf("first tool call name = %q, want echo", result.ToolCalls[0].Name)
			}
		})
	}
}

// TestLive_Claude_StreamingEventsArrive pins that the SSE stream
// produces ModelChunk events and terminates with a TurnDone. Catches
// regressions in the bridle event-shape mapping (e.g. SDK upgrade
// that changes chunk shape, accumulator behaviour change).
func TestLive_Claude_StreamingEventsArrive(t *testing.T) {
	for _, c := range shapeConfigs {
		t.Run(c.name, func(t *testing.T) {
			p, ctx, cancel := liveProvider(t, c)
			defer cancel()
			h := bridle.NewHarness(p)
			rec := &recordingSink{}
			_, err := h.RunTurn(ctx, bridle.TurnRequest{
				Model:       c.model,
				UserMessage: "Write a short two-sentence haiku about debugging.",
				MaxSteps:    1,
			}, noopRunner{}, rec)
			if err != nil {
				t.Fatalf("RunTurn: %v", err)
			}
			if rec.chunkCount == 0 {
				t.Errorf("no ModelChunk events received; streaming broken or text empty")
			}
			if !rec.sawTurnDone {
				t.Errorf("no TurnDone event received; terminal-event mapping broken")
			}
		})
	}
}

// TestLive_Claude_AuthFails is intentionally NOT parameterised — one
// bogus-key request is enough to validate the auth-error mapping;
// hitting every endpoint just wastes API quota with the same outcome.
// Uses ANTHROPIC_API_KEY=bogus against api.anthropic.com to exercise
// the real error path. Skips when no real key is configured (so the
// test infra at least exists in the env).
func TestLive_Claude_AuthFails(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping (need a real env to know live tests are wanted)")
	}
	p := claude.New("sk-ant-bogus-key-that-will-fail-auth")
	h := bridle.NewHarness(p)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       "claude-haiku-4-5",
		UserMessage: "hi",
		MaxSteps:    1,
	}, noopRunner{}, nullSink{})
	if err == nil {
		t.Fatal("expected auth error from bogus key; got nil")
	}
	// Don't assert exact wording — SDK + API error text drifts over
	// time. Just that something error-shaped surfaced.
}

// TestLive_Claude_ModelNotFound similarly validates that a clearly
// invalid model id surfaces as an actionable error. Single shot — same
// rationale as AuthFails.
func TestLive_Claude_ModelNotFound(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping")
	}
	p := claude.New(key)
	h := bridle.NewHarness(p)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       "this-model-definitely-does-not-exist-xyz",
		UserMessage: "hi",
		MaxSteps:    1,
	}, noopRunner{}, nullSink{})
	if err == nil {
		t.Fatal("expected model-not-found error; got nil")
	}
}

// noopRunner is the no-tool path (MaxSteps=1, model emits no tool
// calls). Satisfies bridle.ToolRunner for the live tests.
type noopRunner struct{}

func (noopRunner) Run(_ context.Context, _ bridle.ToolCall) (json.RawMessage, error) {
	return json.RawMessage(`{}`), nil
}

// echoRunner returns the tool call's args verbatim. Lets the model
// see its own input + report it back so we can verify the roundtrip
// actually completed.
type echoRunner struct{}

func (echoRunner) Run(_ context.Context, call bridle.ToolCall) (json.RawMessage, error) {
	if len(call.Args) == 0 {
		return json.RawMessage(`{"echo":""}`), nil
	}
	return call.Args, nil
}

// recordingSink captures whether streaming events arrived in the
// expected shape. Just counts chunks + flags the terminal event;
// fine-grained content assertions would be flaky against real
// providers (output text varies turn to turn).
type recordingSink struct {
	chunkCount  int
	sawTurnDone bool
}

func (r *recordingSink) Emit(ev bridle.Event) {
	switch ev.(type) {
	case bridle.ModelChunk:
		r.chunkCount++
	case bridle.TurnDone:
		r.sawTurnDone = true
	}
}
