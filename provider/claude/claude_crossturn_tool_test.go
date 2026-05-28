package claude_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/claude"
)

// Parallel of openai's cross-turn tool replay tests, against the
// Anthropic wire shape (used by DeepSeek's /anthropic endpoint as
// well as real Anthropic).
//
// Anthropic-shape cross-turn tool replay:
//
//	assistant: { "role":"assistant", "content":[ {"type":"text",...}, {"type":"tool_use","id":...,"name":...,"input":{...}} ] }
//	tool result is a USER msg with a content block:
//	user:      { "role":"user", "content":[ {"type":"tool_result","tool_use_id":...,"content":"..."} ] }
//
// Tests pin the wire body directly via the captured request body —
// no live API call.

type claudeWireRequest struct {
	Model    string                   `json:"model"`
	Messages []map[string]interface{} `json:"messages"`
}

func runClaudeWithCapture(t *testing.T, req bridle.ProviderRequest) claudeWireRequest {
	t.Helper()
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
	if _, err := p.RunTurn(context.Background(), req, nullSink{}); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	raw, _ := h.lastBody.Load().([]byte)
	if len(raw) == 0 {
		t.Fatalf("no request body captured")
	}
	var got claudeWireRequest
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode captured body: %v\nbody=%s", err, string(raw))
	}
	return got
}

func claudeBlocks(t *testing.T, msg map[string]interface{}) []map[string]interface{} {
	t.Helper()
	rawBlocks, ok := msg["content"].([]interface{})
	if !ok {
		t.Fatalf("content is not an array: %+v", msg["content"])
	}
	out := make([]map[string]interface{}, 0, len(rawBlocks))
	for _, b := range rawBlocks {
		out = append(out, b.(map[string]interface{}))
	}
	return out
}

// TestCrossTurnClaude_AssistantToolUseShape — assistant with one
// tool_call renders as an assistant message whose content array
// contains a {type:tool_use, id, name, input} block.
func TestCrossTurnClaude_AssistantToolUseShape(t *testing.T) {
	got := runClaudeWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "do the thing"},
			{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{
				ID: "toolu_abc", Name: "echo", Args: json.RawMessage(`{"text":"ping"}`),
			}}},
			{Role: "tool_result", ToolCallID: "toolu_abc", Content: "ping"},
			{Role: "user", Content: "great"},
		},
	})

	if len(got.Messages) != 4 {
		t.Fatalf("wire messages=%d, want 4", len(got.Messages))
	}

	asst := got.Messages[1]
	if asst["role"] != "assistant" {
		t.Fatalf("msg[1] role=%v, want assistant", asst["role"])
	}
	blocks := claudeBlocks(t, asst)
	var toolUse map[string]interface{}
	for _, b := range blocks {
		if b["type"] == "tool_use" {
			toolUse = b
			break
		}
	}
	if toolUse == nil {
		t.Fatalf("no tool_use block in assistant content: %+v", blocks)
	}
	if toolUse["id"] != "toolu_abc" {
		t.Errorf("tool_use id = %v, want toolu_abc", toolUse["id"])
	}
	if toolUse["name"] != "echo" {
		t.Errorf("tool_use name = %v, want echo", toolUse["name"])
	}
	input, ok := toolUse["input"].(map[string]interface{})
	if !ok {
		t.Fatalf("tool_use input is not a JSON object: %+v", toolUse["input"])
	}
	if input["text"] != "ping" {
		t.Errorf("tool_use.input.text = %v, want ping", input["text"])
	}

	// tool_result rides on a USER message in the anthropic shape.
	toolMsg := got.Messages[2]
	if toolMsg["role"] != "user" {
		t.Fatalf("msg[2] role=%v, want user (anthropic tool_results ride on user msgs)", toolMsg["role"])
	}
	trBlocks := claudeBlocks(t, toolMsg)
	if len(trBlocks) != 1 || trBlocks[0]["type"] != "tool_result" {
		t.Fatalf("msg[2] content[0] = %+v, want tool_result block", trBlocks)
	}
	if trBlocks[0]["tool_use_id"] != "toolu_abc" {
		t.Errorf("tool_result.tool_use_id = %v, want toolu_abc", trBlocks[0]["tool_use_id"])
	}
}

// TestCrossTurnClaude_AssistantMultiToolUseCoalesced — N tool_calls
// on one assistant turn render as N tool_use blocks within ONE
// assistant message (not separate messages).
func TestCrossTurnClaude_AssistantMultiToolUseCoalesced(t *testing.T) {
	got := runClaudeWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "say hi twice"},
			{Role: "assistant", ToolCalls: []bridle.ToolInvocation{
				{ID: "tu_1", Name: "send_chat", Args: json.RawMessage(`{"text":"hi"}`)},
				{ID: "tu_2", Name: "send_chat", Args: json.RawMessage(`{"text":"there"}`)},
			}},
			{Role: "tool_result", ToolCallID: "tu_1", Content: "ok"},
			{Role: "tool_result", ToolCallID: "tu_2", Content: "ok"},
		},
	})

	asst := got.Messages[1]
	blocks := claudeBlocks(t, asst)
	toolUses := 0
	for _, b := range blocks {
		if b["type"] == "tool_use" {
			toolUses++
		}
	}
	if toolUses != 2 {
		t.Errorf("got %d tool_use blocks in one assistant msg, want 2", toolUses)
	}
}

// TestCrossTurnClaude_AssistantTextAndToolUseCoexist — text and
// tool_use blocks render in order in the assistant content array.
func TestCrossTurnClaude_AssistantTextAndToolUseCoexist(t *testing.T) {
	got := runClaudeWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "go"},
			{Role: "assistant", Content: "Calling echo:", ToolCalls: []bridle.ToolInvocation{{
				ID: "tu_x", Name: "echo", Args: json.RawMessage(`{"text":"ping"}`),
			}}},
		},
	})
	asst := got.Messages[1]
	blocks := claudeBlocks(t, asst)
	if len(blocks) < 2 {
		t.Fatalf("want >=2 content blocks; got %d (%+v)", len(blocks), blocks)
	}
	hasText, hasTool := false, false
	for _, b := range blocks {
		switch b["type"] {
		case "text":
			hasText = true
		case "tool_use":
			hasTool = true
		}
	}
	if !hasText || !hasTool {
		t.Errorf("text+tool_use must coexist; hasText=%v hasTool=%v", hasText, hasTool)
	}
}

// TestCrossTurnClaude_LowerRequestCoalescesAssistantSessionEvents —
// integration of lowerRequest + toClaudeMessages: a SessionTail with
// N separate assistant tool_use SessionEvents (RawJSON-bearing)
// collapses into one assistant wire message with N tool_use blocks.
// This is the path nexus's funnel takes when replaying a prior turn.
func TestCrossTurnClaude_LowerRequestCoalescesAssistantSessionEvents(t *testing.T) {
	// claude RawJSON shape: marshaled anthropic.ToolUseBlock.
	mkRaw := func(id, name, input string) json.RawMessage {
		raw, _ := json.Marshal(struct {
			Type  string          `json:"type"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		}{Type: "tool_use", ID: id, Name: name, Input: json.RawMessage(input)})
		return raw
	}

	// Build the SessionTail that the funnel hands to bridle on turn 2.
	tail := []bridle.SessionEvent{
		{Provider: bridle.ProviderClaude, Role: bridle.RoleUser, Content: "say hi twice"},
		{Provider: bridle.ProviderClaude, Role: bridle.RoleAssistant, RawJSON: mkRaw("tu_a", "send_chat", `{"text":"hi"}`)},
		{Provider: bridle.ProviderClaude, Role: bridle.RoleAssistant, RawJSON: mkRaw("tu_b", "send_chat", `{"text":"there"}`)},
		{Provider: bridle.ProviderClaude, Role: bridle.RoleTool, Content: "ok", ToolCallID: "tu_a"},
		{Provider: bridle.ProviderClaude, Role: bridle.RoleTool, Content: "ok", ToolCallID: "tu_b"},
	}

	// Use the harness so lowerRequest fires (the SessionTail path
	// only kicks in via TurnRequest -> harness, not by handing
	// ProviderMessages directly to the provider).
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
	harness := bridle.NewHarness(p)
	_, err := harness.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "test-model",
		SessionTail: tail,
		UserMessage: "thanks",
	}, noopRunner{}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	raw, _ := h.lastBody.Load().([]byte)
	body := string(raw)

	// Expect distinguishing markers from a properly-rebuilt assistant
	// turn with two tool_use blocks + paired tool_result blocks.
	for _, want := range []string{
		`"type":"tool_use"`,
		`"id":"tu_a"`,
		`"id":"tu_b"`,
		`"type":"tool_result"`,
		`"tool_use_id":"tu_a"`,
		`"tool_use_id":"tu_b"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("wire body missing marker %q\nbody=%s", want, body)
		}
	}

	// Verify the assistant content has exactly ONE assistant message
	// holding both tool_use blocks (coalesced), not two separate
	// assistant messages.
	var wire claudeWireRequest
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("decode wire: %v", err)
	}
	assistantCount := 0
	var assistantBlocks []map[string]interface{}
	for _, m := range wire.Messages {
		if m["role"] == "assistant" {
			assistantCount++
			assistantBlocks = claudeBlocks(t, m)
		}
	}
	if assistantCount != 1 {
		t.Errorf("got %d assistant messages on wire, want 1 (coalesced)", assistantCount)
	}
	tu := 0
	for _, b := range assistantBlocks {
		if b["type"] == "tool_use" {
			tu++
		}
	}
	if tu != 2 {
		t.Errorf("coalesced assistant has %d tool_use blocks, want 2", tu)
	}
}

