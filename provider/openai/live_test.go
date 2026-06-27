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

// TestLive_DeepSeek_StrictDegradesToJSONObject (NEX-587) verifies the
// strict-mode degradation end-to-end against the REAL DeepSeek /v1: a
// caller requesting response_format=json_schema strict (which DeepSeek
// rejects with 400 "type unavailable") gets a clean parseable-JSON
// result because the DeepSeek-aware provider degraded the request to
// json_object on the wire — NOT a hard 400. This is the exact
// regression the NEX-297 L2 A/B caught; here we assert the fix holds
// against the live endpoint.
//
// Env-gated on DEEPSEEK_OPENAI_API_KEY; skips cleanly without it.
func TestLive_DeepSeek_StrictDegradesToJSONObject(t *testing.T) {
	key := os.Getenv("DEEPSEEK_OPENAI_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_OPENAI_API_KEY not set; skipping live DeepSeek strict-degrade test")
	}
	p := openai.NewWithBaseURL(key, "https://api.deepseek.com/v1")
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	temp := 0.0
	h := bridle.NewHarness(p)
	result, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:           "deepseek-chat",
		UserMessage:     `Classify this turn as scratch or complete: hello world. Reply with JSON {"class":"...","reason":"..."}.`,
		MaxSteps:        1,
		Temperature:     &temp,
		MaxOutputTokens: 150,
		// Request STRICT json_schema — DeepSeek would 400 this raw; the
		// provider must degrade it to json_object so this succeeds.
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
		t.Fatalf("strict request against DeepSeek must NOT 400 (should degrade to json_object): %v", err)
	}
	var anyJSON map[string]any
	if jerr := json.Unmarshal([]byte(strings.TrimSpace(result.FinalText)), &anyJSON); jerr != nil {
		t.Errorf("degraded json_object response did not parse as JSON: %v\nbody=%s", jerr, result.FinalText)
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

// NEX-340 live: multi-turn against DeepSeek's reasoner model
// (deepseek-v4-pro / deepseek-reasoner). The reasoner emits
// reasoning_content; turn 2 carries it back via SessionTail. Without
// the NEX-340 fix, DeepSeek 400s with "reasoning_content must be
// passed back to the API". With the fix, turn 2 lands cleanly.
//
// Env-gated on DEEPSEEK_REASONER_KEY since this needs a DeepSeek
// account + a reasoner model. Falls through to DEEPSEEK_OPENAI_API_KEY
// if not set separately, since both routes use the same DeepSeek key
// in practice.
func TestLive_OpenAI_DeepSeekReasonerMultiTurn(t *testing.T) {
	key := getenv("DEEPSEEK_REASONER_KEY", "DEEPSEEK_OPENAI_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_REASONER_KEY (or DEEPSEEK_OPENAI_API_KEY) not set; skipping live reasoner test")
	}
	model := getenv("DEEPSEEK_REASONER_MODEL", "")
	if model == "" {
		model = "deepseek-reasoner" // public model id; deepseek-v4-pro is the nexus alias for it
	}
	p := openai.NewWithBaseURL(key, "https://api.deepseek.com/v1")
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	h := bridle.NewHarness(p)

	r1, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       model,
		UserMessage: "Remember the number 42 for me. Reply with just 'ok'.",
		MaxSteps:    1,
	}, noopRunner{}, nullSink{})
	if err != nil {
		t.Fatalf("turn 1 RunTurn: %v", err)
	}
	if r1.FinalText == "" {
		t.Fatalf("turn 1 FinalText empty")
	}

	// Sanity: the reasoner should have emitted reasoning_content
	// somewhere in SessionDelta. If this assertion fails, the model
	// isn't actually a reasoner OR our extraction is broken — either
	// way, the cross-turn test below would silently pass without
	// actually exercising the bug. Belt + braces.
	sawReasoning := false
	for _, e := range r1.SessionDelta {
		if e.ReasoningContent != "" {
			sawReasoning = true
			break
		}
	}
	if !sawReasoning {
		t.Logf("turn 1 SessionDelta carries no reasoning_content — model %q may not be a reasoner; downstream assertion is weaker than intended", model)
	}

	tail := []bridle.SessionEvent{
		{Provider: bridle.ProviderOpenAI, Role: bridle.RoleUser, Content: "Remember the number 42 for me. Reply with just 'ok'."},
	}
	tail = append(tail, r1.SessionDelta...)

	r2, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       model,
		SessionTail: tail,
		UserMessage: "What number did I ask you to remember? Just the digits.",
		MaxSteps:    1,
	}, noopRunner{}, nullSink{})
	if err != nil {
		t.Fatalf("turn 2 RunTurn (the exact path NEX-340 fixes): %v", err)
	}
	if !strings.Contains(r2.FinalText, "42") {
		t.Errorf("turn 2 didn't recall context; FinalText=%q", r2.FinalText)
	}
}

// TestLive_OpenAI_DeepSeekReasonerToolRoundtrip — covers the IN-TURN
// step-2 reasoning_content round-trip. After the reasoner emits a
// tool_call (step 1), bridle reconstructs the assistant message and
// invokes the provider again with the tool_result appended (step 2).
// Without ReasoningContent threaded into that reconstruction, DeepSeek
// rejects step 2 with "The reasoning_content in the thinking mode must
// be passed back to the API". This test fails on the pre-NEX-340-fix
// run.go path and passes once `ReasoningContent: presult.ReasoningContent`
// is added to the assistant reconstruction.
func TestLive_OpenAI_DeepSeekReasonerToolRoundtrip(t *testing.T) {
	key := getenv("DEEPSEEK_REASONER_KEY", "DEEPSEEK_OPENAI_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_REASONER_KEY (or DEEPSEEK_OPENAI_API_KEY) not set; skipping live reasoner tool test")
	}
	model := getenv("DEEPSEEK_REASONER_MODEL", "")
	if model == "" {
		model = "deepseek-reasoner"
	}
	p := openai.NewWithBaseURL(key, "https://api.deepseek.com/v1")
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	h := bridle.NewHarness(p)

	result, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:       model,
		UserMessage: `Call the echo tool with text="ping" and report what you got back.`,
		MaxSteps:    3,
		Tools: []bridle.ToolDef{{
			Name:        "echo",
			Description: "Echo back the input.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`),
		}},
	}, echoRunner{}, nullSink{})
	if err != nil {
		t.Fatalf("reasoner tool roundtrip RunTurn: %v", err)
	}
	if len(result.ToolCalls) == 0 {
		t.Errorf("expected at least one tool call from reasoner; got 0")
	}
}

// TestLive_OpenAI_DeepSeekReasonerSendChatChunking — diagnostic for
// the plumb pathology: aspect emits one send_chat per word of its
// reply, fanning a single intended message into dozens of chat rows.
// The unit-level streaming tests showed bridle's accumulator merges
// chunk deltas correctly. This test asks the actual model the same
// question — given plumb-shaped prompt + the real send_chat tool
// schema, does DeepSeek-v4-pro emit 1 tool_call or N?
//
// Asserts ToolCalls == 1. Failure with N>1 means the model itself
// is parallel-emitting; failure with N==0 means the model bypassed
// the tool. Either way the test surface gives the actual count for
// diagnosis.
//
// The system prompt mirrors plumb's SOUL.md style guidance ("loose,
// sketchy, write the way someone thinks aloud") because that's the
// instruction most likely to interact badly with parallel_tool_calls.
// The send_chat schema is verbatim from nexus/frame/funnel/comms.go.
func TestLive_OpenAI_DeepSeekReasonerSendChatChunking(t *testing.T) {
	key := getenv("DEEPSEEK_REASONER_KEY", "DEEPSEEK_OPENAI_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_REASONER_KEY (or DEEPSEEK_OPENAI_API_KEY) not set; skipping live chunking diagnostic")
	}
	model := getenv("DEEPSEEK_REASONER_MODEL", "")
	if model == "" {
		model = "deepseek-reasoner"
	}
	p := openai.NewWithBaseURL(key, "https://api.deepseek.com/v1")
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	h := bridle.NewHarness(p)

	// Plumb-shaped system prompt (loose style + reply via chat tool).
	// Lifted from the relevant SOUL/PRIMER beats so the model sees
	// the same shape of guidance the real aspect does.
	systemPrompt := `You are plumb, an aspect of Convergence in a multi-agent network.

## Style
Loose. Sketchy. Comfortable being wrong on the way to right. You write in the way someone thinks aloud — if the operator wants polish, that's a downstream pass.

## Reply mechanism
You communicate via the send_chat tool. Every reply you make to the operator is via send_chat.`

	// send_chat tool def, verbatim shape from
	// nexus/frame/funnel/comms.go:218.
	tools := []bridle.ToolDef{{
		Name:        "send_chat",
		Description: "Post a message to the group chat. Use to ask clarifying questions, share status, or reply to an addressed message. Use @<aspect> to mention a specific aspect; replies go to the parent's author plus any explicit @-mentions.",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"content":  {"type": "string", "description": "Message body. Use @aspect to mention."},
				"reply_to": {"type": "integer", "description": "Optional msg_id of the message you're replying to."},
				"topic":    {"type": "string", "description": "Optional topic name for feature-scoped threads."}
			},
			"required": ["content"]
		}`),
	}}

	result, err := h.RunTurn(ctx, bridle.TurnRequest{
		Model:              model,
		AppendSystemPrompt: systemPrompt,
		// A prompt that should provoke a multi-sentence reply. If
		// plumb's tendency is to chunk one-per-word, this is where
		// it surfaces.
		UserMessage: "Hey plumb — give me your two-sentence take on whether worktrees-by-default is a good idea for agent tasks.",
		MaxSteps:    1,
		Tools:       tools,
	}, noopRunner{}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	// Always log what we got — primary use of this test is observing
	// model behavior, the assertion is secondary.
	t.Logf("model emitted %d send_chat tool_call(s) for a single reply prompt:", len(result.ToolCalls))
	for i, tc := range result.ToolCalls {
		t.Logf("  [%d] id=%s args=%s", i, tc.ID, string(tc.Args))
	}

	if len(result.ToolCalls) == 0 {
		t.Fatalf("model bypassed send_chat entirely — FinalText=%q", result.FinalText)
	}
	if len(result.ToolCalls) > 1 {
		t.Errorf("model emitted %d parallel send_chat calls; want 1 (the per-word chunking pathology). See logged args above to inspect the split.", len(result.ToolCalls))
	}
}

func getenv(primary, fallback string) string {
	if v := os.Getenv(primary); v != "" {
		return v
	}
	return os.Getenv(fallback)
}
