package bridle

import (
	"context"
	"encoding/json"
	"errors"
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
	text      string   // emitted as one ModelChunk when non-empty
	chunks    []string // when set, emitted as multiple ModelChunks (text ignored)
	toolCalls []ToolInvocation
	err       error
}

// timingProvider is a scripted direct-API provider double (local
// mirror of fake.Provider). Emits ModelChunk events per the scripted
// step; emits nothing for an empty step.
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
	finalText := step.text
	if len(step.chunks) > 0 {
		finalText = ""
		for _, c := range step.chunks {
			sink.Emit(ModelChunk{Text: c})
			finalText += c
		}
	} else if step.text != "" {
		sink.Emit(ModelChunk{Text: step.text})
	}
	return ProviderResult{
		FinalText:  finalText,
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

// --- Tasks 1+2: total, assembly, startup/stream spans ---

// Pinned clock-call placement for a single text round, no tools
// (clock call #N observes at(N-1)):
//
//	#1 turn start, #2 assembly start, #3 assembly end,
//	#4 provider call start, #5 ModelChunk stamp (first event),
//	#6 provider call end, #7 total. (#8 stamps TurnDone.)
//
// AssemblySecs = 1 step (#2→#3); StartupToFirstEventSecs = 1 step
// (#4→#5); StreamSecs = 1 step (#5→#6); TotalSecs = 6 steps (#1→#7).
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

	if got, want := res.Timing.TotalSecs, steps(6); got != want {
		t.Errorf("TotalSecs = %v; want %v", got, want)
	}
	if len(res.Timing.Rounds) != 1 {
		t.Fatalf("Rounds len = %d; want 1", len(res.Timing.Rounds))
	}
	round := res.Timing.Rounds[0]
	if got, want := round.AssemblySecs, steps(1); got != want {
		t.Errorf("Rounds[0].AssemblySecs = %v; want %v", got, want)
	}
	if got, want := round.StartupToFirstEventSecs, steps(1); got != want {
		t.Errorf("Rounds[0].StartupToFirstEventSecs = %v; want %v", got, want)
	}
	if got, want := round.StreamSecs, steps(1); got != want {
		t.Errorf("Rounds[0].StreamSecs = %v; want %v", got, want)
	}
}

// Every event reaching the caller's sink carries a TS from the
// injected clock. Pinned: ModelChunk stamped by clock call #5 → at(4);
// TurnDone stamped by call #8 → at(7).
func TestRunTurn_TimingEventTimestamps(t *testing.T) {
	clk := newFakeClock()
	h := NewHarness(&timingProvider{steps: []timingStep{{text: "hello"}}})
	h.now = clk.Now
	sink := &sliceSink{}

	if _, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, nil, sink); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(sink.events) != 2 {
		t.Fatalf("events len = %d; want 2 (ModelChunk, TurnDone)", len(sink.events))
	}
	chunk, ok := sink.events[0].(ModelChunk)
	if !ok {
		t.Fatalf("events[0] = %T; want ModelChunk", sink.events[0])
	}
	if !chunk.TS.Equal(at(4)) {
		t.Errorf("ModelChunk.TS = %v; want %v", chunk.TS, at(4))
	}
	done, ok := sink.events[1].(TurnDone)
	if !ok {
		t.Fatalf("events[1] = %T; want TurnDone", sink.events[1])
	}
	if !done.TS.Equal(at(7)) {
		t.Errorf("TurnDone.TS = %v; want %v", done.TS, at(7))
	}
}

// Two ModelChunks in one round: StartupToFirstEventSecs covers
// provider-call→first chunk, StreamSecs covers first chunk→provider
// return. Pinned: #4 call start at(3), #5 chunk1 at(4), #6 chunk2
// at(5), #7 call end at(6) → startup 1 step, stream 2 steps.
func TestRunTurn_TimingStartupStreamSplit(t *testing.T) {
	clk := newFakeClock()
	h := NewHarness(&timingProvider{steps: []timingStep{{chunks: []string{"he", "llo"}}}})
	h.now = clk.Now

	res, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, nil, &sliceSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(res.Timing.Rounds) != 1 {
		t.Fatalf("Rounds len = %d; want 1", len(res.Timing.Rounds))
	}
	round := res.Timing.Rounds[0]
	if got, want := round.StartupToFirstEventSecs, steps(1); got != want {
		t.Errorf("StartupToFirstEventSecs = %v; want %v", got, want)
	}
	if got, want := round.StreamSecs, steps(2); got != want {
		t.Errorf("StreamSecs = %v; want %v", got, want)
	}
	if got, want := res.Timing.TotalSecs, steps(7); got != want {
		t.Errorf("TotalSecs = %v; want %v", got, want)
	}
}

// A round in which the provider emits NO events: the full call
// duration lands in StartupToFirstEventSecs with StreamSecs 0 (there
// was never a first event to split on; the whole wait is pre-stream
// latency). Pinned: #4 call start at(3), #5 call end at(4), #6 total
// at(5) → startup 1 step, stream 0, total 5 steps.
func TestRunTurn_TimingNoEventRound(t *testing.T) {
	clk := newFakeClock()
	h := NewHarness(&timingProvider{steps: []timingStep{{}}})
	h.now = clk.Now

	res, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, nil, &sliceSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(res.Timing.Rounds) != 1 {
		t.Fatalf("Rounds len = %d; want 1", len(res.Timing.Rounds))
	}
	round := res.Timing.Rounds[0]
	if got, want := round.StartupToFirstEventSecs, steps(1); got != want {
		t.Errorf("StartupToFirstEventSecs = %v; want %v (full call duration)", got, want)
	}
	if round.StreamSecs != 0 {
		t.Errorf("StreamSecs = %v; want 0", round.StreamSecs)
	}
	if got, want := res.Timing.TotalSecs, steps(5); got != want {
		t.Errorf("TotalSecs = %v; want %v", got, want)
	}
}

// Provider error: the TurnError event is stamped (it flows through the
// wrapped sink) and the error result still carries the recorded
// timing. Pinned: #4 call start at(3), #5 call end at(4), #6 total
// at(5), #7 stamps TurnError → at(6).
func TestRunTurn_TimingProviderErrorStampsTS(t *testing.T) {
	clk := newFakeClock()
	provErr := errors.New("provider boom")
	h := NewHarness(&timingProvider{steps: []timingStep{{err: provErr}}})
	h.now = clk.Now
	sink := &sliceSink{}

	res, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, nil, sink)
	if !errors.Is(err, provErr) {
		t.Fatalf("err = %v; want %v", err, provErr)
	}

	if len(sink.events) != 1 {
		t.Fatalf("events len = %d; want 1 (TurnError)", len(sink.events))
	}
	te, ok := sink.events[0].(TurnError)
	if !ok {
		t.Fatalf("events[0] = %T; want TurnError", sink.events[0])
	}
	if !te.TS.Equal(at(6)) {
		t.Errorf("TurnError.TS = %v; want %v", te.TS, at(6))
	}
	if got, want := res.Timing.TotalSecs, steps(5); got != want {
		t.Errorf("TotalSecs = %v; want %v", got, want)
	}
	if len(res.Timing.Rounds) != 1 {
		t.Fatalf("Rounds len = %d; want 1", len(res.Timing.Rounds))
	}
	if got, want := res.Timing.Rounds[0].StartupToFirstEventSecs, steps(1); got != want {
		t.Errorf("StartupToFirstEventSecs = %v; want %v", got, want)
	}
	if res.Timing.Rounds[0].StreamSecs != 0 {
		t.Errorf("StreamSecs = %v; want 0", res.Timing.Rounds[0].StreamSecs)
	}
}

// --- Task 3: tool durations + request-size accounting ---

// A turn with one tool call: two provider rounds, each with its own
// assembly/startup/stream spans, plus one ToolTiming bracketing
// executeToolCall. Pinned clock trace (#N observes at(N-1)):
//
//	#1 turn start at(0); #2/#3 round-0 assembly at(1)/at(2);
//	#4 call start at(3), no provider events, #5 call end at(4);
//	#6 tool start at(5) — ToolCallStart stamp at(6), ToolCallResult
//	stamp at(7) — #7 tool end at(8) → Secs = 3 steps;
//	StepBoundary stamp at(9); #8/#9 round-1 assembly at(10)/at(11);
//	#10 call start at(12), chunk stamp at(13), #11 call end at(14);
//	#12 total at(15). (TurnDone stamp at(16).)
func TestRunTurn_TimingToolDurations(t *testing.T) {
	clk := newFakeClock()
	p := &timingProvider{steps: []timingStep{
		{toolCalls: []ToolInvocation{{ID: "1", Name: "echo", Args: json.RawMessage(`{}`)}}},
		{text: "done"},
	}}
	h := NewHarness(p)
	h.now = clk.Now

	res, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
		Tools: []ToolDef{{
			Name:        "echo",
			Description: "echo",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
		MaxSteps: 5,
	}, echoRunner{}, &sliceSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(res.Timing.Tools) != 1 {
		t.Fatalf("Tools len = %d; want 1", len(res.Timing.Tools))
	}
	tool := res.Timing.Tools[0]
	if tool.ID != "1" || tool.Name != "echo" {
		t.Errorf("Tools[0] = {ID:%q Name:%q}; want {ID:\"1\" Name:\"echo\"}", tool.ID, tool.Name)
	}
	if got, want := tool.Secs, steps(3); got != want {
		t.Errorf("Tools[0].Secs = %v; want %v", got, want)
	}

	if len(res.Timing.Rounds) != 2 {
		t.Fatalf("Rounds len = %d; want 2 (one per provider call)", len(res.Timing.Rounds))
	}
	r0, r1 := res.Timing.Rounds[0], res.Timing.Rounds[1]
	if got, want := r0.AssemblySecs, steps(1); got != want {
		t.Errorf("Rounds[0].AssemblySecs = %v; want %v", got, want)
	}
	if got, want := r0.StartupToFirstEventSecs, steps(1); got != want {
		t.Errorf("Rounds[0].StartupToFirstEventSecs = %v; want %v (no-event round: full call)", got, want)
	}
	if r0.StreamSecs != 0 {
		t.Errorf("Rounds[0].StreamSecs = %v; want 0", r0.StreamSecs)
	}
	if got, want := r1.AssemblySecs, steps(1); got != want {
		t.Errorf("Rounds[1].AssemblySecs = %v; want %v", got, want)
	}
	if got, want := r1.StartupToFirstEventSecs, steps(1); got != want {
		t.Errorf("Rounds[1].StartupToFirstEventSecs = %v; want %v", got, want)
	}
	if got, want := r1.StreamSecs, steps(1); got != want {
		t.Errorf("Rounds[1].StreamSecs = %v; want %v", got, want)
	}
	if got, want := res.Timing.TotalSecs, steps(15); got != want {
		t.Errorf("TotalSecs = %v; want %v", got, want)
	}
}

// Per-round request-size accounting: PromptBytes is the marshaled size
// of the assembled preq.Messages; MessageCount/ToolDefCount mirror the
// request the provider actually received that round.
func TestRunTurn_TimingRequestSize(t *testing.T) {
	p := &timingProvider{steps: []timingStep{
		{toolCalls: []ToolInvocation{{ID: "1", Name: "echo", Args: json.RawMessage(`{}`)}}},
		{text: "done"},
	}}
	h := NewHarness(p)
	h.now = newFakeClock().Now

	res, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
		Tools: []ToolDef{{
			Name:        "echo",
			Description: "echo",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
		MaxSteps: 5,
	}, echoRunner{}, &sliceSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(res.Timing.Rounds) != 2 {
		t.Fatalf("Rounds len = %d; want 2", len(res.Timing.Rounds))
	}
	r0, r1 := res.Timing.Rounds[0], res.Timing.Rounds[1]

	// Round 0: exactly the lowered request — one user message.
	wantMsgs, merr := json.Marshal([]ProviderMessage{{Role: "user", Content: "hi"}})
	if merr != nil {
		t.Fatalf("marshal expected messages: %v", merr)
	}
	if got, want := r0.PromptBytes, len(wantMsgs); got != want {
		t.Errorf("Rounds[0].PromptBytes = %d; want %d", got, want)
	}
	if r0.MessageCount != 1 {
		t.Errorf("Rounds[0].MessageCount = %d; want 1", r0.MessageCount)
	}
	if r0.ToolDefCount != 1 {
		t.Errorf("Rounds[0].ToolDefCount = %d; want 1", r0.ToolDefCount)
	}

	// Round 1: user + reconstructed assistant tool_use + tool_result.
	if r1.MessageCount != 3 {
		t.Errorf("Rounds[1].MessageCount = %d; want 3", r1.MessageCount)
	}
	if r1.ToolDefCount != 1 {
		t.Errorf("Rounds[1].ToolDefCount = %d; want 1", r1.ToolDefCount)
	}
	if r1.PromptBytes <= r0.PromptBytes {
		t.Errorf("Rounds[1].PromptBytes = %d; want > Rounds[0].PromptBytes (%d)", r1.PromptBytes, r0.PromptBytes)
	}
}

// A message history that fails json.Marshal (invalid RawMessage args
// injected by a hook) records PromptBytes -1 and never fails the turn.
func TestRunTurn_TimingPromptBytesMarshalError(t *testing.T) {
	h := NewHarness(&timingProvider{steps: []timingStep{{text: "hello"}}})
	h.now = newFakeClock().Now
	h.RegisterBeforeModelCall(func(_ context.Context, in BeforeModelCallCtx) (BeforeModelCallCtx, HookAction, error) {
		in.Request.Messages = append(in.Request.Messages, ProviderMessage{
			Role:      "assistant",
			ToolCalls: []ToolInvocation{{ID: "x", Name: "bad", Args: json.RawMessage(`{not json`)}},
		})
		return in, HookContinue, nil
	})

	res, err := h.RunTurn(context.Background(), TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, nil, &sliceSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.Timing.Rounds) != 1 {
		t.Fatalf("Rounds len = %d; want 1", len(res.Timing.Rounds))
	}
	if got := res.Timing.Rounds[0].PromptBytes; got != -1 {
		t.Errorf("PromptBytes = %d; want -1 on marshal error", got)
	}
	if res.FinalText != "hello" {
		t.Errorf("FinalText = %q; want %q (turn must not fail)", res.FinalText, "hello")
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
