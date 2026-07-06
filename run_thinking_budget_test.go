package bridle

import "testing"

// Extended-thinking budget: lowerRequest must carry
// TurnRequest.ThinkingBudgetTokens onto the lowered ProviderRequest so
// provider/claude can translate it to anthropic.ThinkingConfigParamOfEnabled.
// Mirrors the Temperature/TopK/Seed sampling-knob lowering precedent.
func TestLowerRequest_CarriesThinkingBudgetTokens(t *testing.T) {
	req := TurnRequest{ThinkingBudgetTokens: 2048}
	out := lowerRequest(req)
	if out.ThinkingBudgetTokens != 2048 {
		t.Errorf("ThinkingBudgetTokens = %d, want 2048", out.ThinkingBudgetTokens)
	}
}

func TestLowerRequest_ThinkingBudgetTokensDefaultZero(t *testing.T) {
	req := TurnRequest{}
	out := lowerRequest(req)
	if out.ThinkingBudgetTokens != 0 {
		t.Errorf("ThinkingBudgetTokens = %d, want 0 (unset default)", out.ThinkingBudgetTokens)
	}
}
