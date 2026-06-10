package bridle

import (
	"sync"
	"time"
)

// stampSink decorates the caller's EventSink: stamps TS on every event
// and records the first-event time per provider round for TurnTiming.
// The harness wraps the caller's sink once at runTurn entry and passes
// the wrapped sink to the provider, so provider implementations never
// change.
//
// Threading contract: inner.Emit is called WITHOUT holding mu, so two
// goroutines emitting concurrently may deliver to inner in non-FIFO
// timestamp order. Every current provider serialises its stream
// goroutine before RunTurn returns, so concurrent Emit does not occur
// today; a future parallel-streaming provider must revisit this.
type stampSink struct {
	inner EventSink
	now   func() time.Time

	mu         sync.Mutex
	firstEvent time.Time // zero until the round's first event; reset per round
}

func (s *stampSink) Emit(ev Event) {
	ts := s.now()
	s.mu.Lock()
	if s.firstEvent.IsZero() {
		s.firstEvent = ts
	}
	s.mu.Unlock()
	s.inner.Emit(stampEvent(ev, ts))
}

// roundReset clears the first-event mark; called by runTurn before each
// provider round so harness-emitted events between rounds (tool call
// starts/results, step boundaries) can't masquerade as the next round's
// first provider event.
func (s *stampSink) roundReset() {
	s.mu.Lock()
	s.firstEvent = time.Time{}
	s.mu.Unlock()
}

// takeFirstEvent returns and clears the round's first-event time;
// called by runTurn after the provider round returns. Zero means the
// provider emitted no events this round.
func (s *stampSink) takeFirstEvent() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	first := s.firstEvent
	s.firstEvent = time.Time{}
	return first
}

// stampEvent returns a copy of ev with its TS field set to ts. Events
// are value types, so the emitter's original is never mutated.
func stampEvent(ev Event, ts time.Time) Event {
	switch e := ev.(type) {
	case ModelChunk:
		e.TS = ts
		return e
	case ToolCallStart:
		e.TS = ts
		return e
	case ToolCallResult:
		e.TS = ts
		return e
	case StepBoundary:
		e.TS = ts
		return e
	case TurnDone:
		e.TS = ts
		return e
	case TurnError:
		e.TS = ts
		return e
	case MCPServerFailed:
		e.TS = ts
		return e
	default:
		return ev
	}
}
