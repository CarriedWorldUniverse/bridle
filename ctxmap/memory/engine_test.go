package memory

import (
	"encoding/json"
	"strings"
	"testing"

	distillNewPkg "github.com/CarriedWorldUniverse/bridle/ctxmap/distill"
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
		{Statement: "the render seat moves to ember-node", Kind: "OBSERVED", Source: "user", Force: extractor.ForceDecision, Entities: []string{"ember-node"}},
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
		{Statement: "pool workers are named personality-role", Kind: "OBSERVED", Source: "user", Force: extractor.ForceDecision, Entities: []string{"pool-workers"}},
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

func TestWorkingStateTracksProgress(t *testing.T) {
	st, _ := store.Open(":memory:")
	defer st.Close()
	rend, _ := render.New(st)
	e := New(Config{SessionID: "ws"}, st, rend, nil, nil, nil)
	defer e.Close()

	// disabled by default: no observation, empty block
	e.ObserveTool("write_file", json.RawMessage(`{"path":"a.py","content":"x"}`), `"ok"`)
	if b := e.WorkingMemoryBlock("task"); strings.Contains(b, "Working state") {
		t.Fatal("working-state must be off until enabled")
	}

	e.EnableWorkingState()
	if !e.RefreshMode() {
		t.Fatal("working-state must put the engine in refresh mode")
	}
	e.ObserveTool("read_file", json.RawMessage(`{"path":"src/framing.py"}`), `"..."`)
	e.ObserveTool("write_file", json.RawMessage(`{"path":"src/const.py"}`), `"OK wrote src/const.py"`)
	e.ObserveTool("write_file", json.RawMessage(`{"path":"src/const.py"}`), `"OK wrote src/const.py"`)
	// tool result arrives JSON-encoded — the block must show clean text
	e.ObserveTool("run_command", json.RawMessage(`{"command":"python3 tests/test_codec.py"}`), `"PASS a\nFAIL b: AssertionError\n\n1 FAILED"`)

	b := e.WorkingMemoryBlock("task")
	for _, want := range []string{
		"Working state",
		"src/const.py (2×)",             // edit count tracked
		"python3 tests/test_codec.py",   // last command
		"FAIL b: AssertionError",        // decoded test output (not escaped)
		"1 FAILED",
		"Recent steps:",
	} {
		if !strings.Contains(b, want) {
			t.Fatalf("working-state block missing %q\n---\n%s", want, b)
		}
	}
	if strings.Contains(b, `\n`) {
		t.Fatal("tool output must be decoded, not escaped JSON")
	}
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

func TestForceSemantics(t *testing.T) {
	e, st, prop := rig(t)
	// QUESTION presupposition: dropped entirely
	prop.out = []extractor.FactProposal{{Statement: "the broker is on li1", Kind: "OBSERVED", Source: "user", Force: extractor.ForceQuestion, Entities: []string{"broker"}}}
	e.RecordTurn(1, "why does the broker on li1 keep dropping connections?", "let me look", nil)
	if ids := e.WaitExtraction(); len(ids) != 0 {
		t.Fatal("question presuppositions must not become facts")
	}
	// REPORT: operator trust, PROPOSED entry
	prop.out = []extractor.FactProposal{{Statement: "the build is green", Kind: "OBSERVED", Source: "user", Force: extractor.ForceReport, Entities: []string{"build"}}}
	e.RecordTurn(2, "fyi the build is green", "noted", nil)
	ids := e.WaitExtraction()
	if len(ids) != 1 {
		t.Fatalf("want 1 fact, got %d", len(ids))
	}
	f, _ := st.Get(ids[0])
	if f.Trust != store.TrustOperatorStated || f.Status != store.StatusProposed {
		t.Fatalf("REPORT: want OPERATOR_STATED/PROPOSED, got %s/%s", f.Trust, f.Status)
	}
	// DIRECTIVE: intent is performative -> VERIFIED
	prop.out = []extractor.FactProposal{{Statement: "The operator wants the settlement cap raised to 60", Kind: "PREFERENCE", Source: "user", Force: extractor.ForceDirective, Entities: []string{"settlement-gen"}}}
	e.RecordTurn(3, "raise the settlement cap to 60 please", "on it", nil)
	ids = e.WaitExtraction()
	if len(ids) != 1 {
		t.Fatalf("want 1 fact, got %d", len(ids))
	}
	f, _ = st.Get(ids[0])
	if f.Status != store.StatusVerified {
		t.Fatalf("DIRECTIVE intent is performative: want VERIFIED, got %s", f.Status)
	}
}

type fixedJudge struct{ v extractor.PairVerdict }

func (j *fixedJudge) JudgePair(a, b string) (extractor.PairVerdict, error) { return j.v, nil }

type topicEmbedder struct{}

func (topicEmbedder) Embed(text string) ([]float32, error) {
	// all statements share a topic => always same-topic candidates
	return []float32{1, 0, 0, 0}, nil
}

func TestDirectiveConflictFlagsNotResolves(t *testing.T) {
	e, st, prop := rig(t)
	e2 := e // engine from rig has no embedder/judge; rebuild with them
	_ = e2
	st2, _ := store.Open(":memory:")
	defer st2.Close()
	rend2, _ := render.New(st2)
	prop2 := &fakeProposer{}
	eng := New(Config{SessionID: "test"}, st2, rend2, prop2, topicEmbedder{}, &fixedJudge{v: extractor.PairContradicts})
	defer eng.Close()

	// establish the constraint (operator decision -> VERIFIED)
	prop2.out = []extractor.FactProposal{{Statement: "biome blending must never sample across chunk seams", Kind: "CONSTRAINT", Source: "user", Force: extractor.ForceDecision, Entities: []string{"biome-blending"}}}
	eng.RecordTurn(1, "hard rule: biome blending must never sample across chunk seams", "understood", nil)
	ids := eng.WaitExtraction()
	cid := ids[0]
	rend2.NewEpoch()

	// a directive that violates it: must land as intent, LINKED, constraint NOT retracted
	prop2.out = []extractor.FactProposal{{Statement: "The operator wants biome blending to sample across chunk seams", Kind: "PREFERENCE", Source: "user", Force: extractor.ForceDirective, Entities: []string{"biome-blending"}}}
	eng.RecordTurn(2, "please make biome blending sample across chunk seams to smooth borders", "…", nil)
	ids = eng.WaitExtraction()
	if len(ids) != 1 {
		t.Fatalf("intent fact must land, got %d", len(ids))
	}
	cf, _ := st2.Get(cid)
	if cf.Status == store.StatusRetracted {
		t.Fatal("directive must NOT retract the constraint it conflicts with")
	}
	links, _ := st2.Links(ids[0])
	if len(links[store.LinkContradicts]) == 0 {
		t.Fatal("directive-vs-constraint must be LINKED so the notice fires")
	}
	// and the notice actually renders for the next turn
	b := eng.AssembleBlocks("anything", 3)
	found := false
	for _, n := range b.Notices {
		if strings.Contains(n, cid) || strings.Contains(n, ids[0]) {
			found = true
		}
	}
	if !found {
		t.Fatalf("conflict notice must surface, got %v", b.Notices)
	}
	_ = st
	_ = prop
}

type engFakeSum struct{}

func (engFakeSum) Distill(text, focus string) (string, error) { return "DISTILLED:" + focus, nil }

func TestEngineDistillAndReadRaw(t *testing.T) {
	e, _, _ := rig(t)
	e.SetDistiller(distillNewPkg.New(engFakeSum{}, 50))
	// record a turn so focus is available
	e.RecordTurn(1, "find where AssertFact is defined", "looking", nil)

	raw := strings.Repeat("package store; func AssertFact(...) {...}\n", 20) // > 50 chars
	shown := e.DistillToolResult("Read", raw)
	if !strings.Contains(shown, "DISTILLED:find where AssertFact") {
		t.Fatalf("large tool result must be distilled with task focus, got:\n%s", shown)
	}
	if !strings.Contains(shown, "read_raw") {
		t.Fatal("distilled result must advertise read_raw escalation")
	}
	// read_raw tool present and returns verbatim
	var readRaw *Tool
	for i := range e.Tools() {
		if e.Tools()[i].Name == "read_raw" {
			tl := e.Tools()[i]
			readRaw = &tl
		}
	}
	if readRaw == nil {
		t.Fatal("read_raw tool must be served when a distiller is set")
	}
	i := strings.Index(shown, `handle="`) + len(`handle="`)
	h := shown[i : i+strings.Index(shown[i:], `"`)]
	got := readRaw.Run([]byte(`{"handle":"` + h + `"}`))
	if got != raw {
		t.Fatalf("read_raw must return verbatim raw; got %d chars want %d", len(got), len(raw))
	}
}

func TestNoDistillerNoReadRawTool(t *testing.T) {
	e, _, _ := rig(t)
	for _, tl := range e.Tools() {
		if tl.Name == "read_raw" {
			t.Fatal("read_raw must not be served without a distiller")
		}
	}
	if got := e.DistillToolResult("Read", strings.Repeat("x", 9000)); got != strings.Repeat("x", 9000) {
		t.Fatal("no distiller => pass through")
	}
}
