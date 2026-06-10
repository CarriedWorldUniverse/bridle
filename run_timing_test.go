package bridle

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"
)

// Timing tests live in package bridle (not bridle_test) so they can
// inject the unexported Harness.now clock, mirroring the internal-test
// pattern of run_internal_test.go. The fake/ package can't be imported
// from in-package tests (import cycle: fake imports bridle), so the
// scripted provider/sink doubles below are minimal local mirrors of
// fake.Provider / fake.SliceEventSink.

// fakeClock is a deterministic stepping clock: each Now() call returns
// the current time and then advances 100ms. The first call returns t0
// (Unix 1000). A step counter — never wall time — so tests can pin
// exact span values and catch regressions in Now() call placement.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{t: time.Unix(1000, 0)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.t
	c.t = c.t.Add(100 * time.Millisecond)
	return now
}

// at returns the clock value the Nth Now() call observes (0-based):
// t0 + n*100ms.
func at(n int) time.Time {
	return time.Unix(1000, 0).Add(time.Duration(n) * 100 * time.Millisecond)
}

// ms converts a count of 100ms clock steps into the float64 seconds
// value the implementation computes via Duration.Seconds(), keeping
// exact-equality assertions honest about float representation.
func steps(n int) float64 {
	return (time.Duration(n) * 100 * time.Millisecond).Seconds()
}

// timingStep scripts one provider round for timingProvider.
type timingStep struct {
	text      string
	toolCalls []ToolInvocation
	err       error
}

// timingProvider is a scripted direct-API provider double (local
// mirror of fake.Provider). Emits a single ModelChunk per round when
// text is non-empty; emits nothing otherwise.
type timingProvider struct {
	steps []timingStep
	pos   int
}

func (p *timingProvider) Name() ProviderID { return "fake-timing" }

func (p *timingProvider) Capabilities() ProviderCapabilities {
	return ProviderCapabilities{
		Category:               CategoryDirectAPI,
		SupportsCustomTools:    true,
		SupportsBeforeToolCall: true,
		SupportsAfterToolCall:  true,
		SupportsMCP:            true,
	}
}

func (p *timingProvider) RunTurn(_ context.Context, _ ProviderRequest, sink EventSink) (ProviderResult, error) {
	if p.pos >= len(p.steps) {
		return ProviderResult{StopReason: StopReasonModelDone}, nil
	}
	step := p.steps[p.pos]
	p.pos++
	if step.err != nil {
		return ProviderResult{}, step.err
	}
	if step.text != "" {
		sink.Emit(ModelChunk{Text: step.text})
	}
	return ProviderResult{
		FinalText:  step.text,
		ToolCalls:  step.toolCalls,
		StopReason: StopReasonModelDone,
	}, nil
}

// sliceSink collects events (local mirror of fake.SliceEventSink).
type sliceSink struct {
	events []Event
}

func (s *sliceSink) Emit(e Event) { s.events = append(s.events, e) }

// echoRunner returns a canned payload for any tool call.
type echoRunner struct{}

func (echoRunner) Run(context.Context, ToolCall) (json.RawMessage, error) {
	return json.RawMessage(`"echoed"`), nil
}

// --- Task 1: total + assembly spans ---

// Pinned clock-call placement for a single text round, no tools:
//
//	#1 turn start, #2 assembly start, #3 assembly end, #4 total.
//
// AssemblySecs = 1 step (calls #2→#3); TotalSecs = 3 steps (#1→#4).
func TestRunTurn_TimingTotalAndAssembly(t *testing.T) {
	clk := newFakeClock()
	h := NewHarness(&timingProvider{steps: []timingStep{{text: "hello"}}})
	h.now = clk.Now

	res, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, nil, &sliceSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got, want := res.Timing.TotalSecs, steps(3); got != want {
		t.Errorf("TotalSecs = %v; want %v", got, want)
	}
	if len(res.Timing.Rounds) != 1 {
		t.Fatalf("Rounds len = %d; want 1", len(res.Timing.Rounds))
	}
	if got, want := res.Timing.Rounds[0].AssemblySecs, steps(1); got != want {
		t.Errorf("Rounds[0].AssemblySecs = %v; want %v", got, want)
	}
}

// Default clock: no injection → real time.Now; TotalSecs is recorded
// and positive without any test wiring.
func TestRunTurn_TimingDefaultClock(t *testing.T) {
	h := NewHarness(&timingProvider{steps: []timingStep{{text: "hello"}}})

	res, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, nil, &sliceSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Timing.TotalSecs <= 0 {
		t.Errorf("TotalSecs = %v; want > 0 with the default clock", res.Timing.TotalSecs)
	}
	if len(res.Timing.Rounds) != 1 {
		t.Errorf("Rounds len = %d; want 1", len(res.Timing.Rounds))
	}
}
