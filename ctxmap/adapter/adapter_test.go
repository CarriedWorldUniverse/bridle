package adapter

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/extractor"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/memory"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/render"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/store"
)

type fakeProposer struct{ out []extractor.FactProposal }

func (f *fakeProposer) Propose(extractor.Turn, []extractor.Turn, map[string]string) ([]extractor.FactProposal, error) {
	return f.out, nil
}

type sink struct{}

func (sink) Emit(bridle.Event) {}

func TestAttachEndToEnd(t *testing.T) {
	st, err := store.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	rend, err := render.New(st)
	if err != nil {
		t.Fatal(err)
	}
	// seed a verified fact the model will recall
	fid, err := st.AssertFact(store.Fact{
		Statement: "pool workers are named personality-role", Kind: store.KindObserved,
		Trust: store.TrustOperatorStated, Performative: true, Provenance: []store.Span{{SessionID: "x", Turn: 1, Start: 0, End: 5}},
		Entities: []string{"pool-workers"},
	})
	if err != nil {
		t.Fatal(err)
	}
	rend.NewEpoch() // pull it into the core

	prop := &fakeProposer{out: []extractor.FactProposal{
		{Statement: "the render seat moves to ember-node", Kind: "OBSERVED", Source: "user", Force: extractor.ForceDecision, Entities: []string{"ember-node"}},
	}}
	eng := memory.New(memory.Config{SessionID: "sessA"}, st, rend, prop, nil, nil)
	defer eng.Close()

	// scripted model: step 1 calls recall, step 2 answers
	p := fake.NewProvider(
		fake.Step{ToolCalls: []bridle.ToolInvocation{{ID: "c1", Name: "recall", Args: json.RawMessage(`{"query":"pool workers naming"}`)}}},
		fake.Step{Text: "workers are named personality-role; noting the ember-node move"},
	)
	h := bridle.NewHarness(p)
	detach := Attach(h, eng)
	defer detach()

	res, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		AspectID:    "test",
		Model:       "fake-model",
		UserMessage: "decision: the render seat moves to ember-node. also what did we decide about worker naming?",
	}, fake.NewToolRunner(nil), sink{})
	if err != nil {
		t.Fatal(err)
	}

	// memory blocks reached the provider (LastRequest = the in-loop step-2
	// request; the system prompt persists across steps)
	req := p.LastRequest()
	if !strings.Contains(req.AppendSystemPrompt, "Working memory (automatic)") ||
		!strings.Contains(req.AppendSystemPrompt, "personality-role") {
		t.Fatalf("framing/core missing from system prompt:\n%s", req.AppendSystemPrompt)
	}
	// recall was served from the engine (deny pattern): the step-2 request
	// carries the tool_result containing the seeded fact
	found := false
	for _, m := range req.Messages {
		if m.Role == "tool_result" && strings.Contains(m.Content, fid) {
			found = true
		}
	}
	if !found {
		t.Fatal("recall tool_result with the seeded fact not threaded back")
	}
	if !strings.Contains(res.FinalText, "personality-role") {
		t.Fatalf("final text lost: %q", res.FinalText)
	}

	// OnTurnDone fed extraction: the user's decision landed as a fact
	ids := eng.WaitExtraction()
	if len(ids) != 1 {
		t.Fatalf("want 1 extracted fact, got %d", len(ids))
	}
	f, _ := st.Get(ids[0])
	if f.Trust != store.TrustOperatorStated {
		t.Fatalf("grounded user decision should be OPERATOR_STATED, got %s", f.Trust)
	}
}
