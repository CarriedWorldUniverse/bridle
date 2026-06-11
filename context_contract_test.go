package bridle_test

import (
	"context"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

// TestContextPolicy_PromptOverBudgetWarns — the context contract
// (NEX-581): when the assembled prompt's estimated token count meets or
// exceeds the policy's PromptBudget, the harness emits a
// ContextBudgetWarning carrying the assembled estimate and the budget.
// Engine-agnostic — the fake provider has no context knob, but the
// budget warning rides the existing prompt accounting regardless.
func TestContextPolicy_PromptOverBudgetWarns(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	// A long prompt so the estimator clears a tiny budget.
	big := strings.Repeat("word ", 200)
	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:         "fake-model",
		UserMessage:   big,
		ContextPolicy: bridle.ContextPolicy{PromptBudget: 10},
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var warn *bridle.ContextBudgetWarning
	for _, ev := range sink.Events {
		if w, ok := ev.(bridle.ContextBudgetWarning); ok {
			warn = &w
			break
		}
	}
	if warn == nil {
		t.Fatal("no ContextBudgetWarning emitted for an over-budget prompt")
	}
	if warn.Budget != 10 {
		t.Errorf("warning.Budget = %d, want 10", warn.Budget)
	}
	if warn.Assembled < 10 {
		t.Errorf("warning.Assembled = %d, want >= Budget (10)", warn.Assembled)
	}
	if warn.TS.IsZero() {
		t.Error("warning.TS is zero, want stamped")
	}
}

// TestContextPolicy_UnderBudgetNoWarning — a prompt comfortably under
// budget emits no ContextBudgetWarning.
func TestContextPolicy_UnderBudgetNoWarning(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:         "fake-model",
		UserMessage:   "hi",
		ContextPolicy: bridle.ContextPolicy{PromptBudget: 100000},
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, ev := range sink.Events {
		if _, ok := ev.(bridle.ContextBudgetWarning); ok {
			t.Fatal("ContextBudgetWarning emitted for an under-budget prompt")
		}
	}
}

// TestContextPolicy_ZeroBudgetNeverWarns — a zero PromptBudget (no
// policy) never warns, regardless of prompt size. Current behaviour
// preserved.
func TestContextPolicy_ZeroBudgetNeverWarns(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	big := strings.Repeat("word ", 500)
	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: big,
		// No ContextPolicy — zero value.
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, ev := range sink.Events {
		if _, ok := ev.(bridle.ContextBudgetWarning); ok {
			t.Fatal("ContextBudgetWarning emitted with no PromptBudget policy")
		}
	}
}

// TestContextPolicy_ThreadsToProviderRequest — TargetWindow set on the
// TurnRequest reaches the ProviderRequest the provider sees, so each
// provider can map it to its engine knob.
func TestContextPolicy_ThreadsToProviderRequest(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:         "fake-model",
		UserMessage:   "hi",
		ContextPolicy: bridle.ContextPolicy{TargetWindow: 16384},
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := p.LastRequest().ContextPolicy.TargetWindow; got != 16384 {
		t.Errorf("ProviderRequest.ContextPolicy.TargetWindow = %d, want 16384", got)
	}
}
