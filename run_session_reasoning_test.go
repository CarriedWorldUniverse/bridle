package bridle

import "testing"

// NEX-340 cross-turn: lowerRequest must carry ReasoningContent from
// SessionTail entries onto the rebuilt ProviderMessages. Without this
// the funnel's session-replay path strips the field every Deliberate
// boundary and DeepSeek's reasoner models 400 with "reasoning_content
// must be passed back to the API" on the second turn.
func TestLowerRequest_PreservesReasoningContentFromSessionTail(t *testing.T) {
	req := TurnRequest{
		SessionTail: []SessionEvent{
			{Provider: ProviderOpenAI, Role: RoleAssistant, Content: "Forty-two.",
				ReasoningContent: "Walked 6*7 step by step."},
		},
	}
	out := lowerRequest(req)
	if len(out.Messages) != 1 {
		t.Fatalf("got %d messages, want 1", len(out.Messages))
	}
	m := out.Messages[0]
	if m.ReasoningContent != "Walked 6*7 step by step." {
		t.Errorf("reasoning content lost: %q", m.ReasoningContent)
	}
}

func TestLowerRequest_NoReasoningContentWhenAbsent(t *testing.T) {
	req := TurnRequest{
		SessionTail: []SessionEvent{
			{Provider: ProviderOpenAI, Role: RoleUser, Content: "hi"},
			{Provider: ProviderOpenAI, Role: RoleAssistant, Content: "hello"},
		},
	}
	out := lowerRequest(req)
	for i, m := range out.Messages {
		if m.ReasoningContent != "" {
			t.Errorf("msg %d unexpectedly carried reasoning_content: %q", i, m.ReasoningContent)
		}
	}
}
