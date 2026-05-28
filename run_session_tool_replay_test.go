package bridle

import (
	"encoding/json"
	"testing"
)

// Cross-turn tool-use replay: a prior turn's SessionDelta contains
// N tool_call SessionEvents (one per call, RawJSON-bearing) followed
// by N tool-result events. lowerRequest must coalesce the assistants
// into ONE ProviderMessage with structured ToolCalls and re-attach
// tool_results to their original call IDs — otherwise the rebuilt
// history is invalid (assistant with neither content nor tool_calls,
// tool_results that can't be paired).
func TestLowerRequest_CoalescesAssistantToolCallsAcrossTurn(t *testing.T) {
	tc1 := mustMarshalOpenAIToolCall(t, "call_1", "send_chat", `{"text":"hi"}`)
	tc2 := mustMarshalOpenAIToolCall(t, "call_2", "send_chat", `{"text":"there"}`)

	req := TurnRequest{
		SessionTail: []SessionEvent{
			{Provider: ProviderOpenAI, Role: RoleUser, Content: "say hi twice"},
			{Provider: ProviderOpenAI, Role: RoleAssistant, RawJSON: tc1,
				ReasoningContent: "Need to call send_chat twice."},
			{Provider: ProviderOpenAI, Role: RoleAssistant, RawJSON: tc2},
			{Provider: ProviderOpenAI, Role: RoleTool, Content: "ok", ToolCallID: "call_1"},
			{Provider: ProviderOpenAI, Role: RoleTool, Content: "ok", ToolCallID: "call_2"},
		},
	}
	out := lowerRequest(req)

	if len(out.Messages) != 4 {
		t.Fatalf("got %d messages, want 4 (user + 1 coalesced assistant + 2 tool_results); messages=%+v", len(out.Messages), out.Messages)
	}
	if out.Messages[0].Role != "user" {
		t.Errorf("msg 0 role=%q, want user", out.Messages[0].Role)
	}

	asst := out.Messages[1]
	if asst.Role != "assistant" {
		t.Fatalf("msg 1 role=%q, want assistant", asst.Role)
	}
	if len(asst.ToolCalls) != 2 {
		t.Fatalf("assistant ToolCalls=%d, want 2 (coalesced)", len(asst.ToolCalls))
	}
	if asst.ToolCalls[0].ID != "call_1" || asst.ToolCalls[0].Name != "send_chat" {
		t.Errorf("call 0 = %+v, want call_1/send_chat", asst.ToolCalls[0])
	}
	if asst.ToolCalls[1].ID != "call_2" || asst.ToolCalls[1].Name != "send_chat" {
		t.Errorf("call 1 = %+v, want call_2/send_chat", asst.ToolCalls[1])
	}
	if asst.ReasoningContent != "Need to call send_chat twice." {
		t.Errorf("reasoning_content lost on coalesced assistant: %q", asst.ReasoningContent)
	}

	if out.Messages[2].Role != "tool_result" || out.Messages[2].ToolCallID != "call_1" {
		t.Errorf("msg 2 = %+v, want tool_result/call_1", out.Messages[2])
	}
	if out.Messages[3].Role != "tool_result" || out.Messages[3].ToolCallID != "call_2" {
		t.Errorf("msg 3 = %+v, want tool_result/call_2", out.Messages[3])
	}
}

// Mixed assistant turn: text + tool_call in one logical turn (the
// SessionDelta is two adjacent assistant events). Coalesce keeps the
// text as Content and lifts the tool_call into ToolCalls.
func TestLowerRequest_CoalescesAssistantTextThenToolCall(t *testing.T) {
	tc := mustMarshalOpenAIToolCall(t, "call_x", "echo", `{"text":"ping"}`)
	req := TurnRequest{
		SessionTail: []SessionEvent{
			{Provider: ProviderOpenAI, Role: RoleAssistant, Content: "Calling echo:"},
			{Provider: ProviderOpenAI, Role: RoleAssistant, RawJSON: tc},
		},
	}
	out := lowerRequest(req)
	if len(out.Messages) != 1 {
		t.Fatalf("got %d messages, want 1 coalesced", len(out.Messages))
	}
	m := out.Messages[0]
	if m.Content != "Calling echo:" {
		t.Errorf("content lost: %q", m.Content)
	}
	if len(m.ToolCalls) != 1 || m.ToolCalls[0].Name != "echo" {
		t.Errorf("tool call not lifted: %+v", m.ToolCalls)
	}
}

func mustMarshalOpenAIToolCall(t *testing.T, id, name, args string) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}{
		ID:   id,
		Type: "function",
		Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{Name: name, Arguments: args},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}
