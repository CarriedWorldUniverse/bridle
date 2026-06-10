package bridle_test

import (
	"context"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

// TestRunTurn_ZeroUsageGetsEstimatedFloor — the usage contract
// (NEX-581): a provider that reports no usage must not surface a
// silently-zero turn. After RunTurn, the result's Usage is non-zero and
// flagged Estimated, with plausible counts derived from the prompt and
// response.
func TestRunTurn_ZeroUsageGetsEstimatedFloor(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "this is the model's reply with several words"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "a prompt the model received with enough words to estimate",
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Usage.InputTokens <= 0 {
		t.Errorf("InputTokens = %d, want > 0 (estimated floor)", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens <= 0 {
		t.Errorf("OutputTokens = %d, want > 0 (estimated floor)", result.Usage.OutputTokens)
	}
	if !result.Usage.Estimated {
		t.Error("Usage.Estimated = false, want true when the engine reported no usage")
	}
}

// TestRunTurn_RealUsagePassesThrough — a provider that reports real
// usage has it preserved exactly, with Estimated false. Normalization
// is a no-op on real-usage paths (existing providers unchanged).
func TestRunTurn_RealUsagePassesThrough(t *testing.T) {
	p := fake.NewProvider(fake.Step{
		Text:  "reply",
		Usage: bridle.Usage{InputTokens: 200, OutputTokens: 30},
	})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Usage.InputTokens != 200 || result.Usage.OutputTokens != 30 {
		t.Errorf("real usage mutated: %+v", result.Usage)
	}
	if result.Usage.Estimated {
		t.Error("Usage.Estimated = true, want false when the engine reported real usage")
	}
}
