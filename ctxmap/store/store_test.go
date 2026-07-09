package store

import (
	"strings"
	"testing"
)

func openTest(t *testing.T) *Store {
	t.Helper()
	s, err := Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func span() []Span { return []Span{{SessionID: "s1", Turn: 1, Start: 0, End: 10}} }

func TestP2ConstructionRules(t *testing.T) {
	s := openTest(t)
	// no provenance => rejected
	_, err := s.AssertFact(Fact{Statement: "x is y", Kind: KindObserved, Trust: TrustModelObserved})
	if err == nil || !strings.Contains(err.Error(), "provenance") {
		t.Fatalf("want provenance rejection, got %v", err)
	}
	// DERIVED without parents => rejected
	_, err = s.AssertFact(Fact{Statement: "so z", Kind: KindDerived, Trust: TrustModelDerived, Provenance: span()})
	if err == nil || !strings.Contains(err.Error(), "parents") {
		t.Fatalf("want parents rejection, got %v", err)
	}
	// credential-shaped => rejected
	_, err = s.AssertFact(Fact{Statement: "the token is ghp_abcdefghijklmnopqrstuvwx", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	if err == nil || !strings.Contains(err.Error(), "credential") {
		t.Fatalf("want secret rejection, got %v", err)
	}
}

func TestLifecycleEntryAndPromotion(t *testing.T) {
	s := openTest(t)
	// model-observed enters PROPOSED
	id, err := s.AssertFact(Fact{Statement: "the pod restarted", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	if err != nil {
		t.Fatal(err)
	}
	f, _ := s.Get(id)
	if f.Status != StatusProposed {
		t.Fatalf("want PROPOSED, got %s", f.Status)
	}
	// operator-stated enters VERIFIED
	oid, _ := s.AssertFact(Fact{Statement: "we renamed the node", Kind: KindObserved, Trust: TrustOperatorStated, Performative: true, Provenance: span()})
	of, _ := s.Get(oid)
	if of.Status != StatusVerified {
		t.Fatalf("operator-stated should enter VERIFIED, got %s", of.Status)
	}
	// reuse-confirmation: K distinct render turns promote
	for turn := 1; turn <= ReuseConfirmK; turn++ {
		if err := s.RecordRender(id, turn); err != nil {
			t.Fatal(err)
		}
	}
	f, _ = s.Get(id)
	if f.Status != StatusVerified {
		t.Fatalf("want VERIFIED after %d renders, got %s", ReuseConfirmK, f.Status)
	}
	// same turn repeated does not double-count
	id2, _ := s.AssertFact(Fact{Statement: "another thing happened", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	for i := 0; i < 5; i++ {
		s.RecordRender(id2, 7)
	}
	f2, _ := s.Get(id2)
	if f2.Status != StatusProposed {
		t.Fatalf("repeated same-turn renders must not promote, got %s", f2.Status)
	}
}

func TestContradictedFactDoesNotPromote(t *testing.T) {
	s := openTest(t)
	id, _ := s.AssertFact(Fact{Statement: "cache is enabled", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	other, _ := s.AssertFact(Fact{Statement: "cache is disabled", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	s.Link(other, id, LinkContradicts)
	for turn := 1; turn <= ReuseConfirmK+2; turn++ {
		s.RecordRender(id, turn)
	}
	f, _ := s.Get(id)
	if f.Status != StatusProposed {
		t.Fatalf("contradicted fact must not auto-promote, got %s", f.Status)
	}
}

func TestContradictionTrustRules(t *testing.T) {
	s := openTest(t)
	old, _ := s.AssertFact(Fact{Statement: "results go to sqlite", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	// operator correction beats model fact
	corr, _ := s.AssertFact(Fact{Statement: "results go to postgres", Kind: KindObserved, Trust: TrustOperatorStated, Performative: true, Provenance: span()})
	retracted, err := s.ResolveContradiction(corr, old)
	if err != nil || !retracted {
		t.Fatalf("operator correction should retract, got retracted=%v err=%v", retracted, err)
	}
	of, _ := s.Get(old)
	if of.Status != StatusRetracted {
		t.Fatalf("old fact should be RETRACTED, got %s", of.Status)
	}
	// equal-trust model-vs-model: flag only
	a, _ := s.AssertFact(Fact{Statement: "port is 4000", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	b, _ := s.AssertFact(Fact{Statement: "port is 5000", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	retracted, _ = s.ResolveContradiction(b, a)
	if retracted {
		t.Fatal("equal-trust model contradiction must flag, not retract")
	}
	af, _ := s.Get(a)
	if af.Status == StatusRetracted {
		t.Fatal("flag-only path must not retract")
	}
	// pinned fact: never auto-retract even against operator statement
	p, _ := s.AssertFact(Fact{Statement: "the kernel never allocates mid-tick", Kind: KindConstraint, Trust: TrustModelObserved, Provenance: span()})
	s.Pin(p)
	q, _ := s.AssertFact(Fact{Statement: "the kernel allocates mid-tick", Kind: KindObserved, Trust: TrustOperatorStated, Performative: true, Provenance: span()})
	retracted, _ = s.ResolveContradiction(q, p)
	if retracted {
		t.Fatal("pinned fact must never auto-retract")
	}
}

func TestRetractionCascadesStale(t *testing.T) {
	s := openTest(t)
	parent, _ := s.AssertFact(Fact{Statement: "the loader blocks the frame", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	child, _ := s.AssertFact(Fact{Statement: "so moving loader off-thread fixes stutter", Kind: KindDerived, Trust: TrustModelDerived, Provenance: span(), Parents: []string{parent}})
	s.Retract(parent)
	cf, _ := s.Get(child)
	if !cf.Stale {
		t.Fatal("retracting parent must flag DERIVED child stale")
	}
}

func TestCoreAndAudit(t *testing.T) {
	s := openTest(t)
	v, _ := s.AssertFact(Fact{Statement: "broker lives on li1", Kind: KindObserved, Trust: TrustOperatorStated, Performative: true, Provenance: span()})
	p, _ := s.AssertFact(Fact{Statement: "maybe X", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	core, _ := s.Core()
	if len(core) != 1 || core[0].ID != v {
		t.Fatalf("core must be VERIFIED-only, got %d facts", len(core))
	}
	_ = p
	if issues := s.Audit(); len(issues) != 0 {
		t.Fatalf("clean store should audit clean, got %v", issues)
	}
	// force a live core contradiction and catch it
	v2, _ := s.AssertFact(Fact{Statement: "broker lives on li2", Kind: KindObserved, Trust: TrustOperatorStated, Performative: true, Provenance: span()})
	s.Link(v2, v, LinkContradicts)
	found := false
	for _, i := range s.Audit() {
		if strings.Contains(i, "live contradiction") {
			found = true
		}
	}
	if !found {
		t.Fatal("audit must catch live core contradiction")
	}
}

func TestQueryAndNeighbors(t *testing.T) {
	s := openTest(t)
	a, _ := s.AssertFact(Fact{Statement: "ornith runs vllm behind litellm", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span(), Entities: []string{"ornith", "vllm", "litellm"}})
	b, _ := s.AssertFact(Fact{Statement: "so the bottleneck is litellm routing", Kind: KindDerived, Trust: TrustModelDerived, Provenance: span(), Parents: []string{a}, Entities: []string{"litellm"}})
	got, _ := s.QueryText("vllm", 10, "")
	if len(got) != 1 || got[0].ID != a {
		t.Fatalf("QueryText miss: %d", len(got))
	}
	got, _ = s.QueryEntity("litellm", 10, "")
	if len(got) != 2 {
		t.Fatalf("QueryEntity want 2, got %d", len(got))
	}
	nbrs, _ := s.Neighbors(a, 1)
	if len(nbrs) != 1 || nbrs[0].ID != b {
		t.Fatalf("Neighbors want child, got %d", len(nbrs))
	}
}

func TestCrossSessionVisibility(t *testing.T) {
	s := openTest(t)
	// session A: one VERIFIED (operator) + one PROPOSED (model) fact
	va, _ := s.AssertFact(Fact{Statement: "the broker moved to li1 for good", Kind: KindObserved, Trust: TrustOperatorStated, Performative: true, SessionID: "sessA", Provenance: span(), Entities: []string{"broker"}})
	pa, _ := s.AssertFact(Fact{Statement: "the broker port is probably 7888", Kind: KindObserved, Trust: TrustModelObserved, SessionID: "sessA", Provenance: span(), Entities: []string{"broker"}})

	// session B sees A's VERIFIED fact, not A's PROPOSED
	got, _ := s.QueryEntity("broker", 10, "sessB")
	if len(got) != 1 || got[0].ID != va {
		t.Fatalf("session B should inherit only VERIFIED: got %d facts", len(got))
	}
	// session A still sees both of its own
	got, _ = s.QueryEntity("broker", 10, "sessA")
	if len(got) != 2 {
		t.Fatalf("session A should see its own PROPOSED too: got %d", len(got))
	}
	// unscoped admin view sees everything
	got, _ = s.QueryEntity("broker", 10, "")
	if len(got) != 2 {
		t.Fatalf("admin view should see all: got %d", len(got))
	}
	_ = pa
}


func TestOperatorReportEntersProposed(t *testing.T) {
	s := openTest(t)
	// a world-state report from the operator: top trust, PROPOSED entry
	id, _ := s.AssertFact(Fact{Statement: "the build is green", Kind: KindObserved, Trust: TrustOperatorStated, Performative: false, Provenance: span()})
	f, _ := s.Get(id)
	if f.Status != StatusProposed {
		t.Fatalf("operator REPORT must enter PROPOSED, got %s", f.Status)
	}
	// but it still wins conflicts by trust rank
	m, _ := s.AssertFact(Fact{Statement: "the build is red and failing", Kind: KindObserved, Trust: TrustModelObserved, Provenance: span()})
	retracted, _ := s.ResolveContradiction(id, m)
	if !retracted {
		t.Fatal("operator report must outrank model observation in conflicts")
	}
}
