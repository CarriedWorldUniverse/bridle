package bridle

import "testing"

// Reasoning-effort ladder: lowerRequest must carry TurnRequest.Effort
// onto the lowered ProviderRequest so providers can translate it to
// their own reasoning knob. Mirrors the ThinkingBudgetTokens lowering
// precedent (run_thinking_budget_test.go).
func TestLowerRequest_CarriesEffort(t *testing.T) {
	req := TurnRequest{Effort: "high"}
	out := lowerRequest(req)
	if out.Effort != "high" {
		t.Errorf("Effort = %q, want %q", out.Effort, "high")
	}
}

func TestLowerRequest_EffortDefaultEmpty(t *testing.T) {
	req := TurnRequest{}
	out := lowerRequest(req)
	if out.Effort != "" {
		t.Errorf("Effort = %q, want empty (unset default)", out.Effort)
	}
}
