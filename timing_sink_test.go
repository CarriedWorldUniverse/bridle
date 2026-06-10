package bridle

import (
	"errors"
	"testing"
	"time"
)

// Unit tests for the stampSink decorator. Package bridle (internal) —
// stampSink is unexported by design; the harness is its only caller.

func TestStampSink_StampsEveryEventType(t *testing.T) {
	clk := newFakeClock()
	inner := &sliceSink{}
	s := &stampSink{inner: inner, now: clk.Now}

	events := []Event{
		ModelChunk{Text: "x"},
		ToolCallStart{ID: "1", Name: "echo"},
		ToolCallResult{ID: "1"},
		StepBoundary{Step: 1},
		TurnDone{},
		TurnError{Err: errors.New("boom"), Stage: TurnErrorStageProvider},
	}
	for _, ev := range events {
		s.Emit(ev)
	}

	if len(inner.events) != len(events) {
		t.Fatalf("inner received %d events; want %d", len(inner.events), len(events))
	}
	for i, ev := range inner.events {
		var ts time.Time
		switch e := ev.(type) {
		case ModelChunk:
			ts = e.TS
		case ToolCallStart:
			ts = e.TS
		case ToolCallResult:
			ts = e.TS
		case StepBoundary:
			ts = e.TS
		case TurnDone:
			ts = e.TS
		case TurnError:
			ts = e.TS
		default:
			t.Fatalf("inner.events[%d] = %T; unexpected type", i, ev)
		}
		// Emit call #N stamps at(N-1) — one clock call per event.
		if want := at(i); !ts.Equal(want) {
			t.Errorf("event %d (%T) TS = %v; want %v", i, ev, ts, want)
		}
	}
}

func TestStampSink_FirstEventCaptureAndTake(t *testing.T) {
	clk := newFakeClock()
	s := &stampSink{inner: &sliceSink{}, now: clk.Now}

	s.Emit(ModelChunk{Text: "a"}) // stamped at(0) — first event
	s.Emit(ModelChunk{Text: "b"}) // stamped at(1)

	if first := s.takeFirstEvent(); !first.Equal(at(0)) {
		t.Errorf("takeFirstEvent = %v; want %v (first emit)", first, at(0))
	}
	// take clears the mark.
	if first := s.takeFirstEvent(); !first.IsZero() {
		t.Errorf("second takeFirstEvent = %v; want zero", first)
	}
}

func TestStampSink_RoundResetClearsFirstEvent(t *testing.T) {
	clk := newFakeClock()
	s := &stampSink{inner: &sliceSink{}, now: clk.Now}

	s.Emit(ModelChunk{Text: "round1"}) // at(0)
	s.roundReset()
	if first := s.takeFirstEvent(); !first.IsZero() {
		t.Errorf("takeFirstEvent after roundReset = %v; want zero", first)
	}

	// A fresh round records a fresh first-event mark.
	s.Emit(ModelChunk{Text: "round2"}) // at(1)
	if first := s.takeFirstEvent(); !first.Equal(at(1)) {
		t.Errorf("takeFirstEvent = %v; want %v", first, at(1))
	}
}

func TestStampSink_NoEventsTakeReturnsZero(t *testing.T) {
	s := &stampSink{inner: &sliceSink{}, now: newFakeClock().Now}
	s.roundReset()
	if first := s.takeFirstEvent(); !first.IsZero() {
		t.Errorf("takeFirstEvent with no emits = %v; want zero", first)
	}
}
