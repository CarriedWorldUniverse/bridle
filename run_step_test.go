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

// TestRunStep_RecoversFromPanic proves RunStep has the same recover()
// boundary RunTurn has: a provider panic degrades to a returned error
// rather than crashing the process. The test surviving to completion IS
// the proof — an unrecovered panic would kill the test binary.
func TestRunStep_RecoversFromPanic(t *testing.T) {
	provider := fake.NewProvider(fake.Step{Panic: true})
	h := bridle.NewHarness(provider)
	sink := &fake.SliceEventSink{}

	_, err := h.RunStep(context.Background(), bridle.ProviderRequest{Model: "fake-model"}, sink)
	if err == nil {
		t.Fatal("expected the recovered panic to surface as an error, got nil")
	}
}

// TestRunStep_AppliesUsageFloor proves RunStep applies normalizeUsage
// (the same call RunTurn makes) so a zero-usage turn with non-empty
// FinalText never surfaces silently-zero usage on the Stream path
// (usage.go's "usage required" contract).
func TestRunStep_AppliesUsageFloor(t *testing.T) {
	provider := fake.NewProvider(fake.Step{Text: "a real answer with content"}) // Usage left zero
	h := bridle.NewHarness(provider)
	sink := &fake.SliceEventSink{}

	result, err := h.RunStep(context.Background(), bridle.ProviderRequest{Model: "fake-model"}, sink)
	if err != nil {
		t.Fatalf("RunStep: %v", err)
	}
	if result.Usage.InputTokens == 0 && result.Usage.OutputTokens == 0 {
		t.Errorf("Usage = %+v, want a non-zero estimated floor for a turn with content", result.Usage)
	}
	if !result.Usage.Estimated {
		t.Errorf("Usage.Estimated = false, want true (normalizeUsage should flag the estimated floor)")
	}
}

// TestRunStep_PreservesPartialResultOnRetryError proves RunStep's error
// path returns the accumulated ProviderResult (usage/text from the
// leak-detected round enforceToolCallContract was passed), not a zeroed
// ProviderResult, when the tool-call-contract's retry round itself
// errors — matching RunTurn's "return the partial result" convention.
func TestRunStep_PreservesPartialResultOnRetryError(t *testing.T) {
	garbage := fake.Step{
		Text:  "<|channel|><|message|><|end|>", // strips to nothing -> triggers a retry
		Usage: bridle.Usage{InputTokens: 100, OutputTokens: 50},
	}
	p := fake.NewProvider(garbage, fake.Step{Err: context.DeadlineExceeded})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunStep(context.Background(), bridle.ProviderRequest{Model: "fake-model"}, sink)
	if err == nil {
		t.Fatal("expected the retry round's error to propagate")
	}
	if result.Usage.InputTokens != 100 || result.Usage.OutputTokens != 50 {
		t.Errorf("Usage = %+v, want the partial (garbage-round) usage 100/50 preserved, not zeroed", result.Usage)
	}
}
