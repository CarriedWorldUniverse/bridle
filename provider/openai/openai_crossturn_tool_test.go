package openai_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/openai"
)

// Cross-turn tool replay: the wire shape OpenAI's Chat Completions
// API requires for a prior-turn tool_use round-trip.
//
//	assistant message: { "role":"assistant", "tool_calls":[ { "id":..., "type":"function", "function":{"name":..., "arguments":"..."} } ] }
//	tool result      : { "role":"tool", "tool_call_id":..., "content":"..." }
//
// Both DeepSeek /v1 and vanilla OpenAI enforce this. Verifies the
// wire shape directly instead of standing up a live API call so the
// gate stays deterministic in CI.

// captureRequestBody intercepts ONE request, parses its JSON body,
// then streams the minimal SSE response so RunTurn returns cleanly.
type capturedRequest struct {
	Model    string                   `json:"model"`
	Messages []map[string]interface{} `json:"messages"`
	Tools    []map[string]interface{} `json:"tools,omitempty"`
}

func runWithCapture(t *testing.T, req bridle.ProviderRequest) capturedRequest {
	t.Helper()
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	if _, err := p.RunTurn(context.Background(), req, nullSink{}); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	raw, _ := h.lastBody.Load().([]byte)
	if len(raw) == 0 {
		t.Fatalf("no request body captured")
	}
	var got capturedRequest
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode captured body: %v\nbody=%s", err, string(raw))
	}
	return got
}

// TestCrossTurn_AssistantToolCallShape — when ProviderMessage carries
// ToolCalls, the wire body must render as one assistant message with
// a tool_calls array (objects with id/type/function{name,arguments}),
// NOT as an empty assistant message with a free-floating tool_use
// description. This is what cross-turn replay relies on.
func TestCrossTurn_AssistantToolCallShape(t *testing.T) {
	got := runWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "do the thing"},
			{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{
				ID:   "call_abc",
				Name: "echo",
				Args: json.RawMessage(`{"text":"ping"}`),
			}}},
			{Role: "tool_result", ToolCallID: "call_abc", Content: "ping"},
			{Role: "user", Content: "great, do the next thing"},
		},
	})

	if len(got.Messages) != 4 {
		t.Fatalf("wire messages=%d, want 4; messages=%+v", len(got.Messages), got.Messages)
	}

	asst := got.Messages[1]
	if asst["role"] != "assistant" {
		t.Fatalf("msg[1] role=%v, want assistant", asst["role"])
	}
	calls, ok := asst["tool_calls"].([]interface{})
	if !ok || len(calls) != 1 {
		t.Fatalf("msg[1] missing tool_calls array; got=%+v", asst)
	}
	call0 := calls[0].(map[string]interface{})
	if call0["id"] != "call_abc" || call0["type"] != "function" {
		t.Errorf("tool_calls[0] = %+v, want id=call_abc type=function", call0)
	}
	fn := call0["function"].(map[string]interface{})
	if fn["name"] != "echo" {
		t.Errorf("function.name = %v, want echo", fn["name"])
	}
	if fn["arguments"] != `{"text":"ping"}` {
		t.Errorf("function.arguments = %v, want JSON-string of args", fn["arguments"])
	}

	tool := got.Messages[2]
	if tool["role"] != "tool" {
		t.Fatalf("msg[2] role=%v, want tool", tool["role"])
	}
	if tool["tool_call_id"] != "call_abc" {
		t.Errorf("msg[2] tool_call_id=%v, want call_abc", tool["tool_call_id"])
	}
	if tool["content"] != "ping" {
		t.Errorf("msg[2] content=%v, want ping", tool["content"])
	}
}

// TestCrossTurn_AssistantMultiToolCallCoalesced — strict providers
// (DeepSeek included) require multiple tool_calls in ONE assistant
// message, not separate messages. Plumb's failure case: it called
// send_chat N times in one turn; each landed in SessionDelta as its
// own assistant SessionEvent; without coalescing in lowerRequest we'd
// emit N back-to-back assistant messages and fail wire validation.
// Here we feed ProviderMessage directly with multiple ToolCalls and
// confirm the wire packs them into one message.
func TestCrossTurn_AssistantMultiToolCallCoalesced(t *testing.T) {
	got := runWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "say hi twice"},
			{Role: "assistant", ToolCalls: []bridle.ToolInvocation{
				{ID: "c1", Name: "send_chat", Args: json.RawMessage(`{"text":"hi"}`)},
				{ID: "c2", Name: "send_chat", Args: json.RawMessage(`{"text":"there"}`)},
			}},
			{Role: "tool_result", ToolCallID: "c1", Content: "ok"},
			{Role: "tool_result", ToolCallID: "c2", Content: "ok"},
			{Role: "user", Content: "thanks"},
		},
	})

	asst := got.Messages[1]
	calls := asst["tool_calls"].([]interface{})
	if len(calls) != 2 {
		t.Fatalf("want 2 tool_calls in one assistant message; got %d", len(calls))
	}
	if calls[0].(map[string]interface{})["id"] != "c1" {
		t.Errorf("first call id = %v, want c1", calls[0])
	}
	if calls[1].(map[string]interface{})["id"] != "c2" {
		t.Errorf("second call id = %v, want c2", calls[1])
	}
}

// TestCrossTurn_AssistantReasoningContentWithToolCalls — DeepSeek
// reasoner attaches reasoning_content to the same assistant message
// that emits the tool_call(s). The wire body must carry both fields
// on the same message.
func TestCrossTurn_AssistantReasoningContentWithToolCalls(t *testing.T) {
	got := runWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "do the thing"},
			{Role: "assistant",
				ReasoningContent: "Need to call echo with ping.",
				ToolCalls: []bridle.ToolInvocation{{
					ID: "call_z", Name: "echo", Args: json.RawMessage(`{"text":"ping"}`),
				}},
			},
			{Role: "tool_result", ToolCallID: "call_z", Content: "ping"},
			{Role: "user", Content: "now what"},
		},
	})
	asst := got.Messages[1]
	if _, hasCalls := asst["tool_calls"]; !hasCalls {
		t.Errorf("assistant missing tool_calls: %+v", asst)
	}
	if asst["reasoning_content"] != "Need to call echo with ping." {
		t.Errorf("reasoning_content lost or wrong: %+v", asst["reasoning_content"])
	}
}

// TestCrossTurn_EmptyAssistantWithOnlyReasoningContent_NotMalformed —
// pin the historical breakage: an assistant with reasoning_content
// but no content AND no tool_calls is invalid per OpenAI wire spec
// (DeepSeek rejected with "Invalid assistant message: content or
// tool_calls must be set"). The provider should skip such messages,
// not emit a malformed body. Pre-fix code emitted it; this guards
// against regression.
func TestCrossTurn_EmptyAssistantWithOnlyReasoningContent_NotMalformed(t *testing.T) {
	got := runWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "ping"},
			{Role: "assistant", ReasoningContent: "thinking..."}, // no content, no tool_calls
			{Role: "user", Content: "pong"},
		},
	})
	for i, m := range got.Messages {
		if m["role"] != "assistant" {
			continue
		}
		_, hasContent := m["content"]
		_, hasCalls := m["tool_calls"]
		if !hasContent && !hasCalls {
			t.Errorf("msg[%d] is a malformed assistant (no content + no tool_calls): %+v", i, m)
		}
	}
}

// TestCrossTurn_ToolResultRoleAndKey — pin the wire shape: tool
// results render as role="tool" with a top-level tool_call_id field.
// Anything else (role="tool_result", missing tool_call_id) means the
// API can't pair the result with its call and rejects the request.
func TestCrossTurn_ToolResultRoleAndKey(t *testing.T) {
	got := runWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "go"},
			{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{
				ID: "the_id", Name: "echo", Args: json.RawMessage(`{}`),
			}}},
			{Role: "tool_result", ToolCallID: "the_id", Content: "result"},
		},
	})
	tool := got.Messages[2]
	if tool["role"] != "tool" {
		t.Errorf("tool_result role = %v, want \"tool\" (OpenAI wire name)", tool["role"])
	}
	if tool["tool_call_id"] != "the_id" {
		t.Errorf("tool_call_id = %v, want the_id", tool["tool_call_id"])
	}
}

// TestCrossTurn_BodyShapeMatchesOpenAIContract — full belt-and-braces
// smoke: render the full body to JSON, confirm the shape OpenAI's
// docs prescribe by string-matching distinguishing markers.
func TestCrossTurn_BodyShapeMatchesOpenAIContract(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "go"},
			{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{
				ID: "call_1", Name: "echo", Args: json.RawMessage(`{"text":"ping"}`),
			}}},
			{Role: "tool_result", ToolCallID: "call_1", Content: "ping"},
		},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	raw, _ := h.lastBody.Load().([]byte)
	body := string(raw)
	for _, want := range []string{
		`"role":"assistant"`,
		`"tool_calls":[`,
		`"id":"call_1"`,
		`"type":"function"`,
		`"name":"echo"`,
		`"arguments":"{\"text\":\"ping\"}"`,
		`"role":"tool"`,
		`"tool_call_id":"call_1"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("wire body missing required marker %q\nbody=%s", want, body)
		}
	}
}
