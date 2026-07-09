// Package render turns store facts into the prompt text blocks (spec §6).
//
// The core block is BYTE-STABLE within an epoch (invariant 4): it is rendered
// once when the epoch opens and served verbatim until the next consolidation.
// On a prefix-caching backend that stability is the difference between a warm
// prefix hit and a full cold prefill. All per-turn churn lives in the subgraph
// block, which sits after the core block in the assembled prompt.
package render

import (
	"fmt"
	"sort"
	"strings"

	"github.com/CarriedWorldUniverse/bridle/ctxmap/store"
)

type Renderer struct {
	st       *store.Store
	epoch    int
	coreText string          // frozen at epoch open
	coreIDs  map[string]bool // frozen at epoch open — dedup set for subgraph
}

// New opens epoch 1 over the store's current VERIFIED set.
func New(st *store.Store) (*Renderer, error) {
	r := &Renderer{st: st}
	return r, r.NewEpoch()
}

// NewEpoch re-renders the core block and bumps the epoch — the ONLY operation
// that changes core text (consolidation calls this; it knowingly pays one
// cold prefill).
func (r *Renderer) NewEpoch() error {
	core, err := r.st.Core()
	if err != nil {
		return err
	}
	r.epoch++
	r.coreIDs = map[string]bool{}
	for _, f := range core {
		r.coreIDs[f.ID] = true
	}
	var b strings.Builder
	fmt.Fprintf(&b, "## Working memory — core (epoch %d)\n", r.epoch)
	if len(core) == 0 {
		b.WriteString("(no verified facts yet)\n")
	}
	sort.Slice(core, func(i, j int) bool { return core[i].Created.Before(core[j].Created) })
	for _, f := range core {
		b.WriteString(renderFact(f, false))
	}
	r.coreText = b.String()
	return nil
}

func (r *Renderer) Epoch() int { return r.epoch }

// CoreStale reports whether the live VERIFIED set differs from the frozen
// epoch set — i.e. a consolidation at the next turn boundary would change the
// core block. Cache reseeds are cheap BETWEEN operator turns (operator
// insight 2026-07-09): stability is a within-turn property.
func (r *Renderer) CoreStale() (bool, error) {
	core, err := r.st.Core()
	if err != nil {
		return false, err
	}
	if len(core) != len(r.coreIDs) {
		return true, nil
	}
	for _, f := range core {
		if !r.coreIDs[f.ID] {
			return true, nil
		}
	}
	return false, nil
}

// RenderCore returns the frozen core block. Byte-identical within an epoch.
func (r *Renderer) RenderCore() string { return r.coreText }

// RenderSubgraph renders the per-turn relevant set: seed facts + 1-hop
// neighbors, deduped against the core (core facts are already in context),
// plus contradiction notices for anything touching core/pinned facts.
func (r *Renderer) RenderSubgraph(seeds []*store.Fact) (string, []string) {
	return r.renderSubgraph(seeds, true)
}

// RenderRecall is the pull-tool variant: NO core dedup — a model calling
// recall gets the fact even if it sits in the core block (it asked because
// it missed it there).
func (r *Renderer) RenderRecall(seeds []*store.Fact) (string, []string) {
	return r.renderSubgraph(seeds, false)
}

func (r *Renderer) renderSubgraph(seeds []*store.Fact, dedupCore bool) (string, []string) {
	// dedup against the FROZEN core set: a fact verified mid-epoch is not in
	// the frozen core block, so it must still surface via subgraph/recall.
	coreIDs := r.coreIDs
	if !dedupCore {
		coreIDs = map[string]bool{}
	}
	seen := map[string]bool{}
	var facts []*store.Fact
	add := func(f *store.Fact) {
		if f == nil || seen[f.ID] || coreIDs[f.ID] || f.Status == store.StatusRetracted {
			return
		}
		seen[f.ID] = true
		facts = append(facts, f)
	}
	for _, s := range seeds {
		add(s)
		if nbrs, err := r.st.Neighbors(s.ID, 1); err == nil {
			for _, n := range nbrs {
				add(n)
			}
		}
	}

	var rendered []string // ids actually rendered, for RecordRender bookkeeping
	var b strings.Builder
	b.WriteString("## Working memory — relevant now\n")
	if len(facts) == 0 {
		b.WriteString("(nothing beyond core)\n")
	}
	for _, f := range facts {
		b.WriteString(renderFact(f, true))
		rendered = append(rendered, f.ID)
	}

	// contradiction notices: any CONTRADICTS link touching a core or pinned fact
	notices := map[string]bool{}
	check := append([]*store.Fact{}, seeds...)
	check = append(check, facts...)
	for _, f := range check {
		links, err := r.st.Links(f.ID)
		if err != nil {
			continue
		}
		for _, oid := range links[store.LinkContradicts] {
			of, err := r.st.Get(oid)
			if err != nil {
				continue
			}
			if coreIDs[f.ID] || coreIDs[oid] || f.Pinned || of.Pinned {
				key := orderPair(f.ID, oid)
				if !notices[key] {
					notices[key] = true
					fmt.Fprintf(&b, "- NOTICE: [%s] contradicts [%s]. Verify before relying on either.\n", f.ID, oid)
				}
			}
		}
	}
	return b.String(), rendered
}

func renderFact(f *store.Fact, markStatus bool) string {
	var marks []string
	if f.Kind == store.KindDerived {
		marks = append(marks, "derived from "+strings.Join(f.Parents, ", "))
	}
	if markStatus && f.Status == store.StatusProposed {
		marks = append(marks, "unverified⚠")
	}
	if f.Stale {
		marks = append(marks, "stale⚠")
	}
	switch f.Kind {
	case store.KindConstraint:
		marks = append(marks, "constraint")
	case store.KindPreference:
		marks = append(marks, "preference")
	}
	marks = append(marks, "observed "+f.Created.Format("2006-01-02"))
	return fmt.Sprintf("- [%s] %s (%s)\n", f.ID, f.Statement, strings.Join(marks, "; "))
}

func orderPair(a, b string) string {
	if a < b {
		return a + "|" + b
	}
	return b + "|" + a
}
