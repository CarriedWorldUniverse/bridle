// Package distill is the in-harness distiller tier: the small local model
// compresses large tool results BEFORE they reach the harnessed (expensive,
// remote) model, so that model never spends tokens on raw file dumps, command
// output, or error walls — the noise that degrades a long agentic context.
//
// Safety is the same rule as the whole system: distillation is LOSSY, so it is
// only safe as a CACHE with an escalation path. Every distilled result keeps
// the verbatim raw, retrievable via read_raw, and the distiller FAILS OPEN —
// any summariser error returns the raw unchanged rather than risk dropping a
// load-bearing detail with no way back.
//
// "Local" here means the IN-HARNESS model (the small CPU model that also does
// extraction), NOT the backend being harnessed.
package distill

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
)

// Summarizer compresses text with a task focus. *extractor.Extractor satisfies
// it (behind the ctxmap_llama build tag); tests use a fake.
type Summarizer interface {
	Distill(text, focus string) (string, error)
}

type rawEntry struct {
	toolName string
	raw      string
}

type Distiller struct {
	sum       Summarizer
	threshold int // chars; results at/under this pass through unchanged
	mu        sync.Mutex
	raws      map[string]rawEntry
}

// New builds a distiller. threshold 0 defaults to 1500 chars (~375 tokens):
// below that, distilling costs more than it saves.
func New(sum Summarizer, threshold int) *Distiller {
	if threshold <= 0 {
		threshold = 1500
	}
	return &Distiller{sum: sum, threshold: threshold, raws: map[string]rawEntry{}}
}

// Process compresses a tool result if it exceeds the threshold. Returns the
// text to show the harnessed model. focus is the current task/message so the
// distillation keeps what's relevant. Fails open: any error returns raw.
func (d *Distiller) Process(toolName, raw, focus string) string {
	if d == nil || d.sum == nil || len(raw) <= d.threshold {
		return raw
	}
	summary, err := d.sum.Distill(raw, focus)
	if err != nil || summary == "" {
		return raw // fail open — never drop data on a summariser error
	}
	h := newHandle()
	d.mu.Lock()
	d.raws[h] = rawEntry{toolName: toolName, raw: raw}
	d.mu.Unlock()
	return fmt.Sprintf("%s\n\n[distilled from %s output (%d chars); call read_raw with handle=%q for the verbatim result]",
		summary, toolName, len(raw), h)
}

// Raw returns a stored verbatim result — the escalation path the harnessed
// model uses when it needs certainty the summary can't give.
func (d *Distiller) Raw(handle string) (string, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	e, ok := d.raws[handle]
	if !ok {
		return "", false
	}
	return e.raw, true
}

func newHandle() string {
	b := make([]byte, 5)
	rand.Read(b)
	return "raw_" + hex.EncodeToString(b)
}
