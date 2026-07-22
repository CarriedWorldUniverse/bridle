package bridle_test

import (
	"context"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

// TestHarness_EffortReachesProviderRequest pins the end-to-end path:
// TurnRequest.Effort, set by a real RunTurn caller, must survive the
// harness's lowering (lowerRequest) and arrive on the ProviderRequest
// the provider actually sees — not just asserted against the lowering
// function directly (run_effort_test.go covers that unit in
// isolation), but through the full harness/fake-provider seam a real
// funnel call goes through.
func TestHarness_EffortReachesProviderRequest(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
		Effort:      "medium",
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	if got := p.LastRequest().Effort; got != "medium" {
		t.Errorf("ProviderRequest.Effort = %q, want %q", got, "medium")
	}
}
