package adapter

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/extractor"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/memory"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/render"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/store"
	"github.com/CarriedWorldUniverse/bridle/fake"
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

// newWithinAttachment mirrors Attach's construction for a working-state engine
// (the mode agora runs), returning the attachment so tests can drive the
// BeforeModelCall hook step by step — the placement contract is per-step.
func newWithinAttachment(t *testing.T) *attachment {
	t.Helper()
	st, err := store.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	rend, err := render.New(st)
	if err != nil {
		t.Fatal(err)
	}
	eng := memory.New(memory.Config{SessionID: "s"}, st, rend, nil, nil, nil)
	eng.EnableWorkingState()
	t.Cleanup(eng.Close)
	a := &attachment{eng: eng, tools: map[string]memory.Tool{},
		withinTurn: eng.RefreshMode(), ingest: eng.WithinTurnEnabled(), observe: eng.WorkingStateEnabled()}
	for _, tl := range eng.Tools() {
		a.tools[tl.Name] = tl
	}
	return a
}

// TestWithinMode_BlockAtEndNotSystem: the cache-placement contract. The system
// prompt gets framing ONLY (static prefix); the live working-memory block rides
// as a marker-tagged user message at the END of the message list.
func TestWithinMode_BlockAtEndNotSystem(t *testing.T) {
	a := newWithinAttachment(t)
	a.eng.ObserveTool("run_command", json.RawMessage(`{"command":"go test ./..."}`), `"ok"`)

	req := &bridle.ProviderRequest{
		AppendSystemPrompt: "host system prompt",
		Messages:           []bridle.ProviderMessage{{Role: "user", Content: "do the task"}},
	}
	if _, _, err := a.beforeModelCall(context.Background(), bridle.BeforeModelCallCtx{Request: req, Step: 0}); err != nil {
		t.Fatal(err)
	}

	sys := req.AppendSystemPrompt
	if !strings.Contains(sys, "Working memory (automatic)") || !strings.Contains(sys, "host system prompt") {
		t.Fatalf("system prompt lost framing or host base:\n%s", sys)
	}
	// The block (core header / working state) must NOT be in the system prompt —
	// that placement is what busted the prefix cache on every tool step.
	if strings.Contains(sys, "## Working memory — core") || strings.Contains(sys, "## Working state") {
		t.Fatalf("block leaked into the system prompt:\n%s", sys)
	}
	last := req.Messages[len(req.Messages)-1]
	if last.Role != "user" || !strings.HasPrefix(last.Content, memMsgHeader) {
		t.Fatalf("last message is not the injected memory message: %+v", last)
	}
	if !strings.Contains(last.Content, "go test ./...") {
		t.Fatalf("live working state missing from the injected block:\n%s", last.Content)
	}
	if req.Messages[0].Content != "do the task" {
		t.Fatalf("real user message displaced: %+v", req.Messages[0])
	}
}

// TestWithinMode_SingleInjectionAcrossSteps: across steps the previous
// injection is stripped and re-appended at the new end (exactly one copy), and
// the system prompt stays byte-identical — the prefix-stability regression.
func TestWithinMode_SingleInjectionAcrossSteps(t *testing.T) {
	a := newWithinAttachment(t)
	a.eng.ObserveTool("run_command", json.RawMessage(`{"command":"ls"}`), `"files"`)

	req := &bridle.ProviderRequest{
		AppendSystemPrompt: "host system prompt",
		Messages:           []bridle.ProviderMessage{{Role: "user", Content: "do the task"}},
	}
	if _, _, err := a.beforeModelCall(context.Background(), bridle.BeforeModelCallCtx{Request: req, Step: 0}); err != nil {
		t.Fatal(err)
	}
	sys0 := req.AppendSystemPrompt

	// Simulate run.go's round mutation: assistant tool_use + tool_result are
	// appended AFTER the injected memory message, and the working state moves.
	req.Messages = append(req.Messages,
		bridle.ProviderMessage{Role: "assistant", Content: "running a tool"},
		bridle.ProviderMessage{Role: "tool_result", Content: "tool output", ToolCallID: "c1"},
	)
	a.eng.ObserveTool("write_file", json.RawMessage(`{"path":"x.go"}`), `"ok"`)

	if _, _, err := a.beforeModelCall(context.Background(), bridle.BeforeModelCallCtx{Request: req, Step: 1}); err != nil {
		t.Fatal(err)
	}

	if req.AppendSystemPrompt != sys0 {
		t.Fatalf("system prompt changed between steps (prefix bust):\nstep0: %s\nstep1: %s", sys0, req.AppendSystemPrompt)
	}
	var memIdx []int
	for i, m := range req.Messages {
		if m.Role == "user" && strings.HasPrefix(m.Content, memMsgHeader) {
			memIdx = append(memIdx, i)
		}
	}
	if len(memIdx) != 1 || memIdx[0] != len(req.Messages)-1 {
		t.Fatalf("want exactly one memory message, positioned last; got idx=%v of %d messages", memIdx, len(req.Messages))
	}
	// The conversation ahead of it survived in order.
	want := []string{"do the task", "running a tool", "tool output"}
	for i, w := range want {
		if req.Messages[i].Content != w {
			t.Fatalf("conversation reordered at %d: got %q want %q", i, req.Messages[i].Content, w)
		}
	}
	// The refreshed block carries the NEW working state.
	if !strings.Contains(req.Messages[len(req.Messages)-1].Content, "x.go") {
		t.Fatal("refreshed block missing the new working state")
	}
}
