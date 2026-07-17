package bridle_test

import (
	"context"
	"encoding/json"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

// TestRunStep_SingleRound_NoToolExecution proves RunStep does NOT loop
// or execute tools even when the provider returns ToolCalls — it hands
// them back on the ProviderResult for the caller (agora) to execute.
func TestRunStep_SingleRound_NoToolExecution(t *testing.T) {
	toolCalls := []bridle.ToolInvocation{
		{ID: "call_1", Name: "search", Args: json.RawMessage(`{"q":"x"}`)},
	}
	provider := fake.NewProvider(
		fake.Step{Text: "let me check", ToolCalls: toolCalls},
		// A second scripted step exists ONLY to prove RunStep never
		// consumes it — if RunStep looped like RunTurn, this step would
		// be popped too.
		fake.Step{Text: "should never be reached"},
	)
	h := bridle.NewHarness(provider)
	sink := &fake.SliceEventSink{}

	preq := bridle.ProviderRequest{Model: "fake-model"}
	result, err := h.RunStep(context.Background(), preq, sink)
	if err != nil {
		t.Fatalf("RunStep: %v", err)
	}
	if result.FinalText != "let me check" {
		t.Errorf("FinalText = %q, want %q", result.FinalText, "let me check")
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Name != "search" {
		t.Errorf("ToolCalls = %+v, want the single unexecuted search call", result.ToolCalls)
	}
	if result.ToolCalls[0].Result != nil {
		t.Errorf("ToolCalls[0].Result = %q, want nil — RunStep must not execute tools", result.ToolCalls[0].Result)
	}
	if provider.StepsRemaining() != 1 {
		t.Errorf("StepsRemaining() = %d, want 1 (RunStep must consume exactly ONE scripted step)", provider.StepsRemaining())
	}
}

func TestRunStep_PropagatesProviderError(t *testing.T) {
	provider := fake.NewProvider(fake.Step{Err: context.DeadlineExceeded})
	h := bridle.NewHarness(provider)
	sink := &fake.SliceEventSink{}

	_, err := h.RunStep(context.Background(), bridle.ProviderRequest{Model: "fake-model"}, sink)
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
}
