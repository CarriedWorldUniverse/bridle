// Live integration tests against real OpenAI-shape endpoints
// (NEX-297 Layer 2). Env-var gated — each subtest skips cleanly
// when the relevant API key is absent.
//
// Mirrors the claude live-test surface across the OpenAI Chat
// Completions wire shape. The shape covers a large pool of
// third-party providers that speak it natively: OpenAI itself,
// DeepSeek /v1, Together, Groq, Fireworks, vLLM, local Ollama.
//
// Coverage:
//   - simple text turn
//   - tool roundtrip (function-call fired + result consumed)
//   - streaming event shape
//   - auth fail surfaces cleanly
//   - model-not-found surfaces cleanly
//   - response_format=json_schema strict produces parseable JSON
//     (NEX-300 verification — OpenAI/DeepSeek /v1 should honour it)

package openai_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/openai"
)

type shapeConfig struct {
	name    string
	keyEnv  string
	baseURL string
	model   string
}

var shapeConfigs = []shapeConfig{
	{
		name:    "openai",
		keyEnv:  "OPENAI_API_KEY",
		baseURL: "",
		model:   "gpt-4o-mini",
	},
	{
		name:    "deepseek_v1",
		keyEnv:  "DEEPSEEK_OPENAI_API_KEY",
		baseURL: "https://api.deepseek.com/v1",
		model:   "deepseek-chat",
	},
}

func liveProvider(t *testing.T, c shapeConfig) (*openai.Provider, context.Context, context.CancelFunc) {
	t.Helper()
	key := os.Getenv(c.keyEnv)
	if key == "" {
		t.Skipf("%s not set; skipping live test for shape=%s", c.keyEnv, c.name)
	}
	p := openai.NewWithBaseURL(key, c.baseURL)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	return p, ctx, cancel
}

func TestLive_OpenAI_SimpleTurn(t *testing.T) {
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
				t.Errorf("FinalText empty")
			}
			if result.Usage.InputTokens == 0 {
				t.Errorf("Usage.InputTokens = 0, want non-zero")
			}
			if result.StopReason == "" {
				t.Errorf("StopReason empty")
			}
		})
	}
}

// TestLive_OpenAI_ToolRoundtrip — function-call event extraction +
// tool_result feedback works against the real provider. Both endpoints
// in scope since DeepSeek /v1 is the path we recommend for tool-using
// aspects (NEX-298).
func TestLive_OpenAI_ToolRoundtrip(t *testing.T) {
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
					Description: "Echo back the input.",
					InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`),
				}},
			}, echoRunner{}, nullSink{})
			if err != nil {
				t.Fatalf("RunTurn: %v", err)
			}
			if len(result.ToolCalls) == 0 {
				t.Errorf("expected at least one tool call; got 0")
			}
		})
	}
}

func TestLive_OpenAI_StreamingEventsArrive(t *testing.T) {
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
				t.Errorf("no ModelChunk events received")
			}
			if !rec.sawTurnDone {
				t.Errorf("no TurnDone received")
			}
		})
	}
}

// TestLive_OpenAI_StrictJSONResponseFormat pins the NEX-300 promise:
// response_format=json_schema strict makes the model produce parseable
// JSON matching the schema exactly. **OpenAI-only as of 2026-05-26.**
// DeepSeek's /v1 endpoint returns 400 "This response_format type is
// unavailable now" for the strict variant — discovered by this test
// during NEX-297 L2 verification.
//
// Implication for nexus: CheapModelFilter cannot default to strict
// mode universally; either it skips strict when not-supported, or
// the filter degrades to type=json_object (looser — guarantees JSON
// but not schema match) as the portable default. See nexus follow-up.
func TestLive_OpenAI_StrictJSONResponseFormat(t *testing.T) {
	for _, c := range shapeConfigs {
		t.Run(c.name, func(t *testing.T) {
			// DeepSeek /v1 doesn't support json_schema strict — skip
			// rather than expect failure (the failure mode is well-
			// understood; no value re-firing the 400 each CI run).
			if c.name == "deepseek_v1" {
				t.Skip("DeepSeek /v1 doesn't support response_format=json_schema strict (returns 400 'type unavailable'); see TestLive_OpenAI_JSONObjectResponseFormat for the portable variant")
			}
			p, ctx, cancel := liveProvider(t, c)
			defer cancel()
			temp := 0.0
			h := bridle.NewHarness(p)
			result, err := h.RunTurn(ctx, bridle.TurnRequest{
				Model:           c.model,
				UserMessage:     "Classify this turn as scratch or complete: hello world.",
				MaxSteps:        1,
				Temperature:     &temp,
				MaxOutputTokens: 150,
				ResponseFormat: &bridle.ResponseFormat{
					Type:   "json_schema",
					Name:   "judge_verdict",
					Strict: true,
					Schema: json.RawMessage(`{
						"type": "object",
						"additionalProperties": false,
						"properties": {
							"class":  {"type": "string", "enum": ["complete", "scratch"]},
							"reason": {"type": "string"}
						},
						"required": ["class", "reason"]
					}`),
				},
			}, noopRunner{}, nullSink{})
			if err != nil {
				t.Fatalf("RunTurn: %v", err)
			}
			var verdict struct {
				Class  string `json:"class"`
				Reason string `json:"reason"`
			}
			if err := json.Unmarshal([]byte(strings.TrimSpace(result.FinalText)), &verdict); err != nil {
				t.Fatalf("strict json_schema response did not parse: %v\nbody=%s", err, result.FinalText)
			}
			if verdict.Class != "complete" && verdict.Class != "scratch" {
				t.Errorf("class = %q, want complete or scratch (enum constraint should bind)", verdict.Class)
			}
		})
	}
}

// TestLive_OpenAI_JSONObjectResponseFormat pins the PORTABLE variant
// of response_format — type=json_object guarantees the model returns
// valid JSON (no surrounding prose, no fence) but does NOT constrain
// the JSON's shape. Both OpenAI and DeepSeek /v1 support this.
//
// This is the recommended default for CheapModelFilter going forward
// (NEX-300 follow-up): trade the schema-enforcement guarantee for
// portability. The existing parseJudgeJSON tolerance path handles
// shape mismatches gracefully — the worst-case impact is a parse_
// failure surfacing in logs vs being a non-event, much less bad than
// the all-judge-calls-400 scenario strict mode produces against DeepSeek.
func TestLive_OpenAI_JSONObjectResponseFormat(t *testing.T) {
	for _, c := range shapeConfigs {
		t.Run(c.name, func(t *testing.T) {
			p, ctx, cancel := liveProvider(t, c)
			defer cancel()
			temp := 0.0
			h := bridle.NewHarness(p)
			result, err := h.RunTurn(ctx, bridle.TurnRequest{
				Model:           c.model,
				UserMessage:     `Classify this turn as scratch or complete: hello world. Reply with JSON: {"class":"complete"|"scratch","reason":"..."}`,
				MaxSteps:        1,
				Temperature:     &temp,
				MaxOutputTokens: 150,
				ResponseFormat: &bridle.ResponseFormat{
					Type: "json_object",
				},
			}, noopRunner{}, nullSink{})
			if err != nil {
				t.Fatalf("RunTurn: %v", err)
			}
			// FinalText must parse as JSON (json_object's contract).
			// We don't enforce the shape — that's the trade-off vs
			// strict mode.
			var anyJSON map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(result.FinalText)), &anyJSON); err != nil {
				t.Errorf("json_object response did not parse as JSON: %v\nbody=%s", err, result.FinalText)
			}
		})
	}
}

func TestLive_OpenAI_AuthFails(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set; skipping")
	}
	p := openai.New("sk-bogus-key-that-will-fail-auth")
	h := bridle.NewHarness(p)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       "gpt-4o-mini",
		UserMessage: "hi",
		MaxSteps:    1,
	}, noopRunner{}, nullSink{})
	if err == nil {
		t.Fatal("expected auth error from bogus key; got nil")
	}
}

func TestLive_OpenAI_ModelNotFound(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set; skipping")
	}
	p := openai.New(key)
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

// TestLive_OpenAI_MultiTurnViaSessionTail validates that a second
// Deliberate-equivalent call carrying SessionTail from a prior turn
// gets accepted by the provider. This is the exact path that breaks
// the dMon agents today on the claude-code+DeepSeek-anthropic-shim
// path (NEX-320 thinking-block 400). On the openai+DeepSeek-v1 path
// thinking blocks don't exist, so the cross-turn replay should
// trivially succeed — confirming the recommended migration target.
//
// Test shape: turn 1 establishes a fact; turn 2 (with turn 1's
// assistant reply replayed via SessionTail) asks a follow-up that
// only makes sense if the prior context is preserved.
func TestLive_OpenAI_MultiTurnViaSessionTail(t *testing.T) {
	for _, c := range shapeConfigs {
		t.Run(c.name, func(t *testing.T) {
			p, ctx, cancel := liveProvider(t, c)
			defer cancel()
			h := bridle.NewHarness(p)

			// Turn 1: establish a fact.
			r1, err := h.RunTurn(ctx, bridle.TurnRequest{
				Model:       c.model,
				UserMessage: "Remember this number for later: 42. Reply with just 'ok'.",
				MaxSteps:    1,
			}, noopRunner{}, nullSink{})
			if err != nil {
				t.Fatalf("turn 1 RunTurn: %v", err)
			}
			if r1.FinalText == "" {
				t.Fatalf("turn 1 FinalText empty")
			}

			// Build the cross-Deliberate SessionTail the funnel would
			// hand to the next turn: prior user + prior assistant
			// (lifted from sessionDelta).
			tail := []bridle.SessionEvent{
				{Provider: bridle.ProviderOpenAI, Role: bridle.RoleUser, Content: "Remember this number for later: 42. Reply with just 'ok'."},
			}
			tail = append(tail, r1.SessionDelta...)

			// Turn 2: follow-up that only resolves with prior context.
			r2, err := h.RunTurn(ctx, bridle.TurnRequest{
				Model:       c.model,
				SessionTail: tail,
				UserMessage: "What number did I ask you to remember? Reply with just the digits.",
				MaxSteps:    1,
			}, noopRunner{}, nullSink{})
			if err != nil {
				t.Fatalf("turn 2 RunTurn: %v", err)
			}
			if !strings.Contains(r2.FinalText, "42") {
				t.Errorf("turn 2 did not recall context; FinalText=%q", r2.FinalText)
			}
		})
	}
}

type noopRunner struct{}

func (noopRunner) Run(_ context.Context, _ bridle.ToolCall) (json.RawMessage, error) {
	return json.RawMessage(`{}`), nil
}

type echoRunner struct{}

func (echoRunner) Run(_ context.Context, call bridle.ToolCall) (json.RawMessage, error) {
	if len(call.Args) == 0 {
		return json.RawMessage(`{"echo":""}`), nil
	}
	return call.Args, nil
}

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
