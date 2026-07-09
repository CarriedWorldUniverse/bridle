package memory

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/CarriedWorldUniverse/bridle/ctxmap/extractor"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/render"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/store"
)

type fakeProposer struct{ out []extractor.FactProposal }

func (f *fakeProposer) Propose(extractor.Turn, []extractor.Turn, map[string]string) ([]extractor.FactProposal, error) {
	return f.out, nil
}

func rig(t *testing.T) (*Engine, *store.Store, *fakeProposer) {
	t.Helper()
	st, err := store.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	rend, err := render.New(st)
	if err != nil {
		t.Fatal(err)
	}
	prop := &fakeProposer{}
	e := New(Config{SessionID: "test"}, st, rend, prop, nil, nil)
	t.Cleanup(func() { e.Close(); st.Close() })
	return e, st, prop
}

func TestAssembleRecordExtractLoop(t *testing.T) {
	e, st, prop := rig(t)
	prop.out = []extractor.FactProposal{
		{Statement: "the render seat moves to ember-node", Kind: "OBSERVED", Source: "user", Entities: []string{"ember-node"}},
	}
	b := e.AssembleBlocks("decision: the render seat moves to ember-node", 1)
	if !strings.Contains(b.Framing, "automatic") || !strings.Contains(b.Core, "Working memory — core") {
		t.Fatal("framing/core blocks missing")
	}
	e.RecordTurn(1, "decision: the render seat moves to ember-node", "noted", b.RenderedIDs)
	ids := e.WaitExtraction()
	if len(ids) != 1 {
		t.Fatalf("want 1 fact, got %d", len(ids))
	}
	f, _ := st.Get(ids[0])
	// grounded user fact => operator trust, VERIFIED
	if f.Trust != store.TrustOperatorStated || f.Status != store.StatusVerified {
		t.Fatalf("want OPERATOR_STATED/VERIFIED, got %s/%s", f.Trust, f.Status)
	}
	// next assembly auto-consolidates: the fact joins the core
	b2 := e.AssembleBlocks("anything", 2)
	if !strings.Contains(b2.Core, "ember-node") {
		t.Fatal("auto-consolidation must pull verified fact into core")
	}
}

func TestToolsRecallAndInspect(t *testing.T) {
	e, st, prop := rig(t)
	prop.out = []extractor.FactProposal{
		{Statement: "pool workers are named personality-role", Kind: "OBSERVED", Source: "user", Entities: []string{"pool-workers"}},
	}
	e.RecordTurn(1, "note: pool workers are named personality-role", "got it", nil)
	ids := e.WaitExtraction()

	var recall, inspect Tool
	for _, tl := range e.Tools() {
		switch tl.Name {
		case "recall":
			recall = tl
		case "inspect":
			inspect = tl
		}
	}
	out := recall.Run(json.RawMessage(`{"query":"pool workers naming"}`))
	if !strings.Contains(out, ids[0]) {
		t.Fatalf("recall must return the fact:\n%s", out)
	}
	out = inspect.Run(json.RawMessage(`{"fact_id":"` + ids[0] + `"}`))
	if !strings.Contains(out, "EVIDENCE") || !strings.Contains(out, "personality-role") {
		t.Fatalf("inspect must show evidence:\n%s", out)
	}
	_ = st
}

func TestExtractionDisabledDegradesGracefully(t *testing.T) {
	st, _ := store.Open(":memory:")
	defer st.Close()
	rend, _ := render.New(st)
	e := New(Config{SessionID: "ro"}, st, rend, nil, nil, nil) // no proposer
	defer e.Close()
	e.RecordTurn(1, "hello", "world", nil)
	if ids := e.WaitExtraction(); len(ids) != 0 {
		t.Fatal("no proposer must mean no extraction")
	}
}
