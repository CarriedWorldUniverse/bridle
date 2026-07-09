package render

import (
	"strings"
	"testing"

	"github.com/CarriedWorldUniverse/bridle/ctxmap/store"
)

func setup(t *testing.T) (*store.Store, *Renderer) {
	t.Helper()
	st, err := store.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	r, err := New(st)
	if err != nil {
		t.Fatal(err)
	}
	return st, r
}

func sp() []store.Span { return []store.Span{{SessionID: "s", Turn: 1, Start: 0, End: 5}} }

func TestCoreByteStableWithinEpoch(t *testing.T) {
	st, r := setup(t)
	first := r.RenderCore()
	// new VERIFIED fact lands mid-epoch — core text must NOT change
	st.AssertFact(store.Fact{Statement: "broker on li1", Kind: store.KindObserved, Trust: store.TrustOperatorStated, Provenance: sp()})
	if r.RenderCore() != first {
		t.Fatal("core text changed within an epoch (invariant 4 violated)")
	}
	// epoch bump picks it up
	if err := r.NewEpoch(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.RenderCore(), "broker on li1") {
		t.Fatal("new epoch must include newly verified fact")
	}
	if r.Epoch() != 2 {
		t.Fatalf("epoch want 2, got %d", r.Epoch())
	}
}

func TestSubgraphMarksAndDedup(t *testing.T) {
	st, r := setup(t)
	vid, _ := st.AssertFact(store.Fact{Statement: "kernel never allocates mid-tick", Kind: store.KindConstraint, Trust: store.TrustOperatorStated, Provenance: sp()})
	pid, _ := st.AssertFact(store.Fact{Statement: "loader blocks the frame", Kind: store.KindObserved, Trust: store.TrustModelObserved, Provenance: sp()})
	did, _ := st.AssertFact(store.Fact{Statement: "so loader must move off-thread", Kind: store.KindDerived, Trust: store.TrustModelDerived, Provenance: sp(), Parents: []string{pid}})
	r.NewEpoch() // core now holds vid

	vf, _ := st.Get(vid)
	pf, _ := st.Get(pid)
	text, rendered := r.RenderSubgraph([]*store.Fact{vf, pf})
	// core fact deduped out of subgraph
	if strings.Contains(text, vid) {
		t.Fatal("core fact must not repeat in subgraph")
	}
	// proposed fact marked unverified; derived neighbor pulled in and marked
	if !strings.Contains(text, "unverified⚠") {
		t.Fatal("PROPOSED fact must be marked unverified")
	}
	if !strings.Contains(text, did) || !strings.Contains(text, "derived from "+pid) {
		t.Fatalf("derived neighbor missing or unmarked:\n%s", text)
	}
	// bookkeeping list covers rendered non-core facts only
	for _, id := range rendered {
		if id == vid {
			t.Fatal("rendered list must exclude core facts")
		}
	}
}

func TestContradictionNoticeOnCoreFact(t *testing.T) {
	st, r := setup(t)
	cid, _ := st.AssertFact(store.Fact{Statement: "pod layout is single", Kind: store.KindObserved, Trust: store.TrustOperatorStated, Provenance: sp()})
	r.NewEpoch()
	nid, _ := st.AssertFact(store.Fact{Statement: "pod layout is split", Kind: store.KindObserved, Trust: store.TrustModelObserved, Provenance: sp()})
	st.Link(nid, cid, store.LinkContradicts)

	nf, _ := st.Get(nid)
	text, _ := r.RenderSubgraph([]*store.Fact{nf})
	if !strings.Contains(text, "NOTICE") || !strings.Contains(text, cid) {
		t.Fatalf("expected contradiction notice touching core fact:\n%s", text)
	}
}
