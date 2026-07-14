// Metering for the in-harness models. The token counters a host harness gets
// from its backend provider cover ONLY the remote model; the local Qwen
// extractor/judge/distiller calls never pass through a Provider, so their cost
// is otherwise invisible. This meter makes it countable, so a run can report a
// TOTAL cost (backend + internal), not just the remote bill. Input tokens are
// estimated from prompt bytes (chars/4) — output tokens and wall time are exact.
package extractor

import (
	"sync/atomic"
	"time"
)

// Stat is one category's aggregate (extract / kind / pair / distill).
type Stat struct {
	Calls       int
	OutTokens   int           // exact (llama reports generated tokens)
	InTokensEst int           // estimate: prompt bytes / 4
	Wall        time.Duration // exact
}

type catStat struct {
	calls, outTok, inChars, nanos atomic.Int64
}

func (c *catStat) add(outTok, inChars int, d time.Duration) {
	c.calls.Add(1)
	c.outTok.Add(int64(outTok))
	c.inChars.Add(int64(inChars))
	c.nanos.Add(int64(d))
}

func (c *catStat) snap() Stat {
	return Stat{
		Calls:       int(c.calls.Load()),
		OutTokens:   int(c.outTok.Load()),
		InTokensEst: int(c.inChars.Load() / 4),
		Wall:        time.Duration(c.nanos.Load()),
	}
}

// Meter accumulates internal-model cost by category. Safe for concurrent use.
type Meter struct {
	extract, kind, pair, distill catStat
}

// MeterReport is a point-in-time snapshot of internal-model cost.
type MeterReport struct {
	Extract, Kind, Pair, Distill Stat
}

// Total sums all categories.
func (r MeterReport) Total() Stat {
	sum := func(ss ...Stat) Stat {
		var t Stat
		for _, s := range ss {
			t.Calls += s.Calls
			t.OutTokens += s.OutTokens
			t.InTokensEst += s.InTokensEst
			t.Wall += s.Wall
		}
		return t
	}
	return sum(r.Extract, r.Kind, r.Pair, r.Distill)
}

func (m *Meter) report() MeterReport {
	return MeterReport{
		Extract: m.extract.snap(),
		Kind:    m.kind.snap(),
		Pair:    m.pair.snap(),
		Distill: m.distill.snap(),
	}
}
