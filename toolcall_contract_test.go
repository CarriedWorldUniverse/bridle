package bridle_test

import (
	"context"
	"encoding/json"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

// countEvents returns how many ToolCallRepaired events with the given Stage
// were emitted.
func countRepairedStage(events []bridle.Event, stage string) int {
	n := 0
	for _, e := range events {
		if r, ok := e.(bridle.ToolCallRepaired); ok && r.Stage == stage {
			n++
		}
	}
	return n
}

func anyRepaired(events []bridle.Event) bool {
	for _, e := range events {
		if _, ok := e.(bridle.ToolCallRepaired); ok {
			return true
		}
	}
	return false
}

// --- Clean providers unaffected: no repair/retry path triggered ---

func TestToolCallContract_CleanProviderUnaffected(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "hello, the deploy is green"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "status?",
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinalText != "hello, the deploy is green" {
		t.Errorf("FinalText = %q; want unchanged", result.FinalText)
	}
	if anyRepaired(sink.Events) {
		t.Errorf("clean provider triggered a ToolCallRepaired event")
	}
	// No extra provider round consumed.
	if p.StepsRemaining() != 0 {
		t.Errorf("StepsRemaining = %d; want 0 (no retry)", p.StepsRemaining())
	}
}

// --- Repair-in-place: leak cleaned without a retry round (strict, clean repair) ---

func TestToolCallContract_RepairsLeakedTokensInPlace(t *testing.T) {
	// Leaked tokens around clean text → repaired, no retry needed.
	p := fake.NewProvider(fake.Step{
		Text: "<|channel|>final<|message|>The deploy is green.<|end|>",
	})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "status?",
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinalText != "The deploy is green." {
		t.Errorf("FinalText = %q; want cleaned 'The deploy is green.'", result.FinalText)
	}
	if countRepairedStage(sink.Events, "detected") != 1 {
		t.Errorf("want 1 detected event, got %d", countRepairedStage(sink.Events, "detected"))
	}
	if countRepairedStage(sink.Events, "repaired") != 1 {
		t.Errorf("want 1 repaired event, got %d", countRepairedStage(sink.Events, "repaired"))
	}
	if countRepairedStage(sink.Events, "retried") != 0 {
		t.Errorf("want 0 retried events (repair sufficed), got %d", countRepairedStage(sink.Events, "retried"))
	}
	if p.StepsRemaining() != 0 {
		t.Errorf("StepsRemaining = %d; want 0 (no retry round)", p.StepsRemaining())
	}
}

// --- Extract a tool-call-as-text into a structured call, then execute it ---

func TestToolCallContract_ExtractsAndExecutesToolCallFromText(t *testing.T) {
	// Round 1: the engine leaked a tool call as literal text instead of a
	// structured call. Repair extracts it; the harness then EXECUTES it.
	// Round 2 (the model's response to the tool result): clean final text.
	p := fake.NewProvider(
		fake.Step{Text: `{"name":"echo","arguments":{"msg":"hi"}}`},
		fake.Step{Text: "done"},
	)
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"echo": {{Result: json.RawMessage(`"echoed"`)}},
	})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:    "fake-model",
		Tools:    []bridle.ToolDef{toolDef("echo")},
		MaxSteps: 5,
	}, runner, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d; want 1 (extracted+executed)", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "echo" {
		t.Errorf("extracted call name = %q; want echo", result.ToolCalls[0].Name)
	}
	if result.FinalText != "done" {
		t.Errorf("FinalText = %q; want 'done'", result.FinalText)
	}
	if countRepairedStage(sink.Events, "repaired") != 1 {
		t.Errorf("want 1 repaired (extracted) event, got %d", countRepairedStage(sink.Events, "repaired"))
	}
}

// --- Retry: garbage on round 1, clean on retry. Usage/timing counted ONCE ---

func TestToolCallContract_RetryGarbageThenClean_UsageCountedOnce(t *testing.T) {
	// Round 1 emits unrepairable garble WITH usage. The retry round emits a
	// clean turn WITH its own usage. The retry REPLACES the bad round: only
	// the retried usage is counted, and only one RoundTiming entry exists.
	garbage := fake.Step{
		Text:  "<|channel|><|message|><|end|>", // strips to nothing → not clean → retry
		Usage: bridle.Usage{InputTokens: 100, OutputTokens: 50},
	}
	clean := fake.Step{
		Text:  "All good, deploy is green.",
		Usage: bridle.Usage{InputTokens: 110, OutputTokens: 20},
	}
	p := fake.NewProvider(garbage, clean)
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "status?",
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinalText != "All good, deploy is green." {
		t.Errorf("FinalText = %q; want the retried clean text", result.FinalText)
	}
	// Usage counted ONCE — the retried round's, not the sum of both.
	if result.Usage.InputTokens != 110 || result.Usage.OutputTokens != 20 {
		t.Errorf("Usage = %+v; want only the retried round (110/20), not summed", result.Usage)
	}
	// Exactly one RoundTiming entry — the retry overwrote, not appended.
	if len(result.Timing.Rounds) != 1 {
		t.Errorf("Timing.Rounds len = %d; want 1 (retry replaces the bad round)", len(result.Timing.Rounds))
	}
	if countRepairedStage(sink.Events, "retried") != 1 {
		t.Errorf("want exactly 1 retry, got %d", countRepairedStage(sink.Events, "retried"))
	}
	// Both scripted steps consumed (garbage + retry).
	if p.StepsRemaining() != 0 {
		t.Errorf("StepsRemaining = %d; want 0 (garbage + retry consumed)", p.StepsRemaining())
	}
}

// --- Retry capped at one: garbage both times → flagged cleaned text, no loop ---

func TestToolCallContract_GarbageBothTimes_FlaggedAfterOneRetry(t *testing.T) {
	// Both rounds leak unrepairably-to-clean (strip to nothing). Round 1
	// triggers a retry; round 2 is ALSO garbage but the contract must NOT
	// loop — exactly one retry, then surface the flagged best-effort text.
	g1 := fake.Step{Text: "<|channel|><|message|><|end|>"}
	g2 := fake.Step{Text: "<|channel|><|message|><|end|>"}
	p := fake.NewProvider(g1, g2)
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "status?",
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Both rounds strip to nothing → best-effort surfaced text is empty,
	// flagged.
	if result.FinalText != "" {
		t.Errorf("FinalText = %q; want empty best-effort cleaned text", result.FinalText)
	}
	// Exactly one retry — no infinite loop.
	if got := countRepairedStage(sink.Events, "retried"); got != 1 {
		t.Errorf("retried count = %d; want exactly 1 (capped)", got)
	}
	// The second (post-retry) garble surfaces flagged, not retried again.
	if got := countRepairedStage(sink.Events, "flagged"); got != 1 {
		t.Errorf("flagged count = %d; want exactly 1 (post-retry garble)", got)
	}
	if p.StepsRemaining() != 0 {
		t.Errorf("StepsRemaining = %d; want 0 (only two rounds total)", p.StepsRemaining())
	}
}

// --- Strictness knob: tolerant accepts repaired text without a retry ---

func TestToolCallContract_TolerantAcceptsRepairWithoutRetry(t *testing.T) {
	// Unrepairable-to-clean garble that under STRICT would retry. Under
	// tolerant, bridle surfaces the best-effort repaired text with NO retry.
	p := fake.NewProvider(fake.Step{Text: "<|channel|><|message|><|end|>"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:              "fake-model",
		UserMessage:        "status?",
		ToolCallStrictness: bridle.ToolCallStrictnessTolerant,
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if countRepairedStage(sink.Events, "retried") != 0 {
		t.Errorf("tolerant must not retry; got %d retries", countRepairedStage(sink.Events, "retried"))
	}
	if countRepairedStage(sink.Events, "flagged") != 1 {
		t.Errorf("tolerant want 1 flagged event, got %d", countRepairedStage(sink.Events, "flagged"))
	}
	if p.StepsRemaining() != 0 {
		t.Errorf("StepsRemaining = %d; want 0 (no retry round)", p.StepsRemaining())
	}
	// best-effort cleaned text surfaced (empty here, but the field is set).
	if result.FinalText != "" {
		t.Errorf("FinalText = %q; want empty cleaned best-effort", result.FinalText)
	}
}

// --- Strictness knob: strict retries on a degraded result ---

func TestToolCallContract_StrictRetriesOnDegraded(t *testing.T) {
	// Default (strict) on the same unrepairable round → a retry IS made.
	p := fake.NewProvider(
		fake.Step{Text: "<|channel|><|message|><|end|>"},
		fake.Step{Text: "recovered cleanly"},
	)
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "status?",
		// ToolCallStrictness empty → default repair-then-retry.
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if countRepairedStage(sink.Events, "retried") != 1 {
		t.Errorf("strict (default) want 1 retry, got %d", countRepairedStage(sink.Events, "retried"))
	}
	if result.FinalText != "recovered cleanly" {
		t.Errorf("FinalText = %q; want retried clean text", result.FinalText)
	}
}
