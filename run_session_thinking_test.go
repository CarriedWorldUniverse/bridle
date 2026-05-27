package bridle

import (
	"testing"
)

// NEX-320 cross-turn: lowerRequest must carry ThinkingBlocks from
// SessionTail entries onto the rebuilt ProviderMessages. Without this
// the funnel's session-replay path strips the blocks every Deliberate
// boundary and Anthropic 400s.
func TestLowerRequest_PreservesThinkingBlocksFromSessionTail(t *testing.T) {
	tb := []ThinkingBlock{{Type: "thinking", Thinking: "preserved", Signature: "sig-pres"}}
	req := TurnRequest{
		SessionTail: []SessionEvent{
			{Provider: ProviderClaude, Role: RoleAssistant, Content: "hi", ThinkingBlocks: tb},
		},
	}
	out := lowerRequest(req)
	if len(out.Messages) != 1 {
		t.Fatalf("got %d messages, want 1", len(out.Messages))
	}
	m := out.Messages[0]
	if m.Role != "assistant" || m.Content != "hi" {
		t.Errorf("role/content lost: %+v", m)
	}
	if len(m.ThinkingBlocks) != 1 || m.ThinkingBlocks[0].Signature != "sig-pres" {
		t.Errorf("thinking blocks lost: %+v", m.ThinkingBlocks)
	}
}

// SessionEvents without thinking blocks (user messages, old logs, non-
// thinking-mode assistant turns) lower as before — no spurious empty
// ThinkingBlocks slice on the wire (the field is omitempty in the
// session JSON but ProviderMessage carrying nil → toClaudeMessages
// short-circuits).
func TestLowerRequest_NoThinkingBlocksWhenAbsent(t *testing.T) {
	req := TurnRequest{
		SessionTail: []SessionEvent{
			{Provider: ProviderClaude, Role: RoleUser, Content: "hello"},
			{Provider: ProviderClaude, Role: RoleAssistant, Content: "hi back"},
		},
	}
	out := lowerRequest(req)
	if len(out.Messages) != 2 {
		t.Fatalf("got %d messages, want 2", len(out.Messages))
	}
	for i, m := range out.Messages {
		if len(m.ThinkingBlocks) != 0 {
			t.Errorf("message %d unexpectedly has thinking blocks: %+v", i, m.ThinkingBlocks)
		}
	}
}
