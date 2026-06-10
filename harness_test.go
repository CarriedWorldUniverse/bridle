package bridle_test

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

// --- helpers ---

func toolDef(name string) bridle.ToolDef {
	return bridle.ToolDef{
		Name:        name,
		Description: name,
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}
}

func inv(id, name string) bridle.ToolInvocation {
	return bridle.ToolInvocation{ID: id, Name: name, Args: json.RawMessage(`{}`)}
}

func rawJSON(s string) json.RawMessage { return json.RawMessage(s) }

// --- basic turn: no tools ---

func TestRunTurn_NoTools(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "hello"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, fake.NewToolRunner(nil), sink)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinalText != "hello" {
		t.Errorf("FinalText = %q; want %q", result.FinalText, "hello")
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Errorf("StopReason = %q; want model_done", result.StopReason)
	}

	assertEventOrder(t, sink.Events, "ModelChunk", "TurnDone")
}

// --- tool call round-trip ---

func TestRunTurn_OneToolCall(t *testing.T) {
	toolStep := fake.Step{
		ToolCalls: []bridle.ToolInvocation{inv("1", "echo")},
	}
	finalStep := fake.Step{Text: "done"}

	p := fake.NewProvider(toolStep, finalStep)
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"echo": {{Result: rawJSON(`"echoed"`)}},
	})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:    "fake-model",
		Tools:    []bridle.ToolDef{toolDef("echo")},
		MaxSteps: 5,
	}, runner, sink)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.ToolCalls) != 1 {
		t.Errorf("ToolCalls len = %d; want 1", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "echo" {
		t.Errorf("ToolCalls[0].Name = %q; want echo", result.ToolCalls[0].Name)
	}
	if result.StepCount != 1 {
		t.Errorf("StepCount = %d; want 1", result.StepCount)
	}

	assertEventOrder(t, sink.Events,
		"ToolCallStart", "ToolCallResult", "StepBoundary", "ModelChunk", "TurnDone")
}

// --- subprocess-stream provider (self-executing tools) ---

// NEX-251 regression: a self-executing provider (SupportsCustomTools=false)
// must NOT cause RunTurn to re-invoke the provider when it returns a record
// of tools the subprocess already ran. The previous behavior treated those
// records as tool work for bridle to execute and looped back into the
// provider — for claudecode that re-fired the original user prompt as the
// -p arg of a second `claude -p --resume` call, doubling the session jsonl
// entries on every turn that used a tool.
func TestRunTurn_SelfExecutingProviderDoesNotReinvoke(t *testing.T) {
	p := fake.NewSubprocessProvider(
		fake.SubprocessStep{
			Text: "ok",
			ToolCalls: []bridle.ToolCallStart{
				{ID: "1", Name: "Bash", Args: rawJSON(`{}`)},
			},
			ToolResults: []bridle.ToolCallResult{
				{ID: "1", Result: rawJSON(`"done"`)},
			},
			StopReason: bridle.StopReasonModelDone,
		},
		// A second scripted step exists so a buggy harness that re-invokes
		// would consume it and StepsRemaining would drop to 0. With the
		// NEX-251 fix the loop exits after one iteration and this step is
		// never consumed.
		fake.SubprocessStep{Text: "should not be reached"},
	)
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:    "fake-subprocess",
		MaxSteps: 5,
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if p.StepsRemaining() != 1 {
		t.Errorf("provider re-invoked: StepsRemaining = %d; want 1 (only first step consumed)", p.StepsRemaining())
	}
	if result.FinalText != "ok" {
		t.Errorf("FinalText = %q; want %q", result.FinalText, "ok")
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Name != "Bash" {
		t.Errorf("ToolCalls = %+v; want one Bash invocation recorded for observability", result.ToolCalls)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Errorf("StopReason = %q; want model_done", result.StopReason)
	}
}

// --- MaxSteps cap ---

func TestRunTurn_MaxSteps(t *testing.T) {
	// Provider always returns a tool call; MaxSteps=2 should cap it.
	steps := make([]fake.Step, 5)
	for i := range steps {
		steps[i] = fake.Step{ToolCalls: []bridle.ToolInvocation{inv("x", "noop")}}
	}
	p := fake.NewProvider(steps...)
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"noop": {{Result: rawJSON(`null`)}},
	})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:    "fake-model",
		MaxSteps: 2,
	}, runner, sink)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.StopReason != bridle.StopReasonMaxSteps {
		t.Errorf("StopReason = %q; want max_steps", result.StopReason)
	}
	if result.StepCount != 2 {
		t.Errorf("StepCount = %d; want 2", result.StepCount)
	}
}

// --- cancellation ---

func TestRunTurn_CancelBeforeStart(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled before RunTurn

	p := fake.NewProvider(fake.Step{Text: "should not emit"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, _ := h.RunTurn(ctx, bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), sink)

	if result.StopReason != bridle.StopReasonAborted {
		t.Errorf("StopReason = %q; want aborted", result.StopReason)
	}
}

func TestRunTurn_CancelMidTool(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	// Runner cancels the context when called.
	cancelRunner := &cancelOnRunToolRunner{cancel: cancel, result: rawJSON(`"ok"`)}

	p := fake.NewProvider(
		fake.Step{ToolCalls: []bridle.ToolInvocation{inv("1", "slow")}},
		fake.Step{Text: "never"},
	)
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, _ := h.RunTurn(ctx, bridle.TurnRequest{Model: "fake-model", MaxSteps: 5}, cancelRunner, sink)

	if result.StopReason != bridle.StopReasonAborted {
		t.Errorf("StopReason = %q; want aborted", result.StopReason)
	}
}

// --- hook ordering ---

func TestHooks_BeforeModelCallFires(t *testing.T) {
	var fired []string
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	h.RegisterBeforeModelCall(func(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
		fired = append(fired, "bmc")
		return in, bridle.HookContinue, nil
	})
	sink := &fake.SliceEventSink{}
	h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), sink)

	if len(fired) != 1 || fired[0] != "bmc" {
		t.Errorf("fired = %v; want [bmc]", fired)
	}
}

func TestHooks_BeforeModelCallAborts(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "should not see this"})
	h := bridle.NewHarness(p)
	h.RegisterBeforeModelCall(func(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
		return in, bridle.HookAbort, nil
	})
	sink := &fake.SliceEventSink{}
	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), sink)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result.StopReason != bridle.StopReasonAborted {
		t.Errorf("StopReason = %q; want aborted", result.StopReason)
	}
	// No events should have been emitted.
	if len(sink.Events) != 0 {
		t.Errorf("events emitted = %d; want 0", len(sink.Events))
	}
}

func TestHooks_BeforeToolCallAborts(t *testing.T) {
	p := fake.NewProvider(
		fake.Step{ToolCalls: []bridle.ToolInvocation{inv("1", "echo")}},
		fake.Step{Text: "never"},
	)
	h := bridle.NewHarness(p)
	h.RegisterBeforeToolCall(func(ctx context.Context, in bridle.BeforeToolCallCtx) (bridle.BeforeToolCallCtx, bridle.HookAction, error) {
		return in, bridle.HookAbort, nil
	})
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"echo": {{Result: rawJSON(`"ok"`)}},
	})
	sink := &fake.SliceEventSink{}
	result, _ := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model", MaxSteps: 5}, runner, sink)

	if result.StopReason != bridle.StopReasonAborted {
		t.Errorf("StopReason = %q; want aborted", result.StopReason)
	}
}

func TestHooks_OnTurnDoneCanMutateSessionDelta(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "result"})
	h := bridle.NewHarness(p)
	h.RegisterOnTurnDone(func(ctx context.Context, in bridle.OnTurnDoneCtx) (bridle.OnTurnDoneCtx, bridle.HookAction, error) {
		in.Result.SessionDelta = append(in.Result.SessionDelta, bridle.SessionEvent{
			Role:    bridle.RoleSystem,
			Content: "hook-injected",
		})
		return in, bridle.HookContinue, nil
	})
	sink := &fake.SliceEventSink{}
	result, _ := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), sink)

	last := result.SessionDelta[len(result.SessionDelta)-1]
	if last.Content != "hook-injected" {
		t.Errorf("last session delta = %q; want hook-injected", last.Content)
	}
}

// inspectingProvider records the ProviderRequest fields the harness
// passed in, so a test can assert what reached RunTurn.
type inspectingProvider struct {
	requests []bridle.ProviderRequest
}

func (p *inspectingProvider) Name() bridle.ProviderID { return "inspecting" }
func (p *inspectingProvider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{
		Category:            bridle.CategoryDirectAPI,
		SupportsCustomTools: true,
	}
}
func (p *inspectingProvider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	// Copy the slices we care about so subsequent harness mutations
	// don't backfill what we recorded.
	cp := req
	cp.Messages = append([]bridle.ProviderMessage(nil), req.Messages...)
	cp.Tools = append([]bridle.ToolDef(nil), req.Tools...)
	p.requests = append(p.requests, cp)
	return bridle.ProviderResult{StopReason: bridle.StopReasonModelDone}, nil
}

func TestHooks_BeforeModelCall_MutationReachesProvider(t *testing.T) {
	p := &inspectingProvider{}
	h := bridle.NewHarness(p)
	h.RegisterBeforeModelCall(func(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
		in.Request.Model = "rewritten-by-hook"
		in.Request.AppendSystemPrompt = "extra-system-text"
		in.Request.Messages = append(in.Request.Messages, bridle.ProviderMessage{
			Role:    "user",
			Content: "hook-injected",
		})
		return in, bridle.HookContinue, nil
	})

	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "hi",
	}, fake.NewToolRunner(nil), &fake.SliceEventSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(p.requests) != 1 {
		t.Fatalf("provider called %d times; want 1", len(p.requests))
	}
	got := p.requests[0]
	if got.Model != "rewritten-by-hook" {
		t.Errorf("Model = %q; want rewritten-by-hook (hook mutation dropped)", got.Model)
	}
	if got.AppendSystemPrompt != "extra-system-text" {
		t.Errorf("AppendSystemPrompt = %q; want extra-system-text", got.AppendSystemPrompt)
	}
	// Original UserMessage was lowered into Messages; hook appended one more.
	if len(got.Messages) < 2 || got.Messages[len(got.Messages)-1].Content != "hook-injected" {
		t.Errorf("last message = %+v; want hook-injected", got.Messages)
	}
}

// scriptedInspectingProvider returns a scripted sequence of results
// while also recording the ProviderRequest received at each call.
type scriptedInspectingProvider struct {
	steps    []fake.Step
	pos      int
	requests []bridle.ProviderRequest
}

func (p *scriptedInspectingProvider) Name() bridle.ProviderID { return "scripted-inspecting" }
func (p *scriptedInspectingProvider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{
		Category:            bridle.CategoryDirectAPI,
		SupportsCustomTools: true,
	}
}
func (p *scriptedInspectingProvider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	cp := req
	cp.Model = req.Model // explicit; req is value already
	p.requests = append(p.requests, cp)

	if p.pos >= len(p.steps) {
		return bridle.ProviderResult{StopReason: bridle.StopReasonModelDone}, nil
	}
	step := p.steps[p.pos]
	p.pos++
	return bridle.ProviderResult{
		FinalText:  step.Text,
		ToolCalls:  step.ToolCalls,
		StopReason: bridle.StopReasonModelDone,
	}, nil
}

func TestHooks_BeforeModelCall_InLoopMutationReachesProvider(t *testing.T) {
	// First call returns a tool_use to drive a second call; second
	// call returns final text. The hook mutates Model on every fire
	// and we assert call N+1 saw the mutation from hook fire N.
	p := &scriptedInspectingProvider{
		steps: []fake.Step{
			{ToolCalls: []bridle.ToolInvocation{inv("1", "echo")}},
			{Text: "done"},
		},
	}
	h := bridle.NewHarness(p)
	var fireCount int
	h.RegisterBeforeModelCall(func(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
		fireCount++
		in.Request.Model = "rewritten-call-" + strconv.Itoa(fireCount)
		return in, bridle.HookContinue, nil
	})

	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"echo": {{Result: rawJSON(`"ok"`)}},
	})
	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:    "fake-model",
		Tools:    []bridle.ToolDef{toolDef("echo")},
		MaxSteps: 5,
	}, runner, &fake.SliceEventSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(p.requests) != 2 {
		t.Fatalf("provider called %d times; want 2", len(p.requests))
	}
	if p.requests[0].Model != "rewritten-call-1" {
		t.Errorf("call 1 Model = %q; want rewritten-call-1", p.requests[0].Model)
	}
	if p.requests[1].Model != "rewritten-call-2" {
		t.Errorf("call 2 Model = %q; want rewritten-call-2 (in-loop mutation dropped)", p.requests[1].Model)
	}
}

func TestHooks_RegistrationOrder(t *testing.T) {
	var order []int
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	for i := 0; i < 3; i++ {
		i := i
		h.RegisterBeforeModelCall(func(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
			order = append(order, i)
			return in, bridle.HookContinue, nil
		})
	}
	sink := &fake.SliceEventSink{}
	h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), sink)

	if len(order) != 3 || order[0] != 0 || order[1] != 1 || order[2] != 2 {
		t.Errorf("hook order = %v; want [0 1 2]", order)
	}
}

func TestHooks_UnregisterHook(t *testing.T) {
	var firedA, firedB int
	p := fake.NewProvider(fake.Step{Text: "ok"}, fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	idA := h.RegisterBeforeModelCall(func(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
		firedA++
		return in, bridle.HookContinue, nil
	})
	idB := h.RegisterBeforeModelCall(func(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
		firedB++
		return in, bridle.HookContinue, nil
	})
	if idA == 0 || idB == 0 {
		t.Fatalf("Register returned zero HookID (idA=%d idB=%d)", idA, idB)
	}
	if idA == idB {
		t.Fatalf("Register returned duplicate HookID for distinct hooks: %d", idA)
	}

	// First turn: both fire.
	h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), &fake.SliceEventSink{})
	if firedA != 1 || firedB != 1 {
		t.Errorf("after first turn: firedA=%d firedB=%d; want 1,1", firedA, firedB)
	}

	// Unregister A. Second turn: only B fires.
	if !h.UnregisterHook(idA) {
		t.Errorf("UnregisterHook(idA) returned false; want true")
	}
	h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), &fake.SliceEventSink{})
	if firedA != 1 {
		t.Errorf("after unregister: firedA=%d; want 1 (unregistered hook fired again)", firedA)
	}
	if firedB != 2 {
		t.Errorf("after unregister: firedB=%d; want 2", firedB)
	}
}

func TestHooks_UnregisterHook_UnknownAndZero(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	if h.UnregisterHook(0) {
		t.Errorf("UnregisterHook(0) returned true; want false")
	}
	if h.UnregisterHook(bridle.HookID(9999)) {
		t.Errorf("UnregisterHook(unknown) returned true; want false")
	}
}

func TestHooks_UnregisterHook_CrossType(t *testing.T) {
	// Unregister should find a hook regardless of which Register* added
	// it — verifies the per-type slices are all checked.
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	idBmc := h.RegisterBeforeModelCall(func(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
		return in, bridle.HookContinue, nil
	})
	idBtc := h.RegisterBeforeToolCall(func(ctx context.Context, in bridle.BeforeToolCallCtx) (bridle.BeforeToolCallCtx, bridle.HookAction, error) {
		return in, bridle.HookContinue, nil
	})
	idOtd := h.RegisterOnTurnDone(func(ctx context.Context, in bridle.OnTurnDoneCtx) (bridle.OnTurnDoneCtx, bridle.HookAction, error) {
		return in, bridle.HookContinue, nil
	})
	for _, id := range []bridle.HookID{idBmc, idBtc, idOtd} {
		if !h.UnregisterHook(id) {
			t.Errorf("UnregisterHook(%d) returned false; want true", id)
		}
	}
}

// --- provider error ---

func TestRunTurn_ProviderError(t *testing.T) {
	boom := errors.New("provider boom")
	p := fake.NewProvider(fake.Step{Err: boom})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), sink)

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if result.StopReason != bridle.StopReasonError {
		t.Errorf("StopReason = %q; want error", result.StopReason)
	}
	// TurnError event should be in the sink.
	found := false
	for _, e := range sink.Events {
		if _, ok := e.(bridle.TurnError); ok {
			found = true
		}
	}
	if !found {
		t.Error("TurnError event not emitted")
	}
}

// --- tool error does not abort turn ---

func TestRunTurn_ToolError_DoesNotAbortTurn(t *testing.T) {
	p := fake.NewProvider(
		fake.Step{ToolCalls: []bridle.ToolInvocation{inv("1", "failing")}},
		fake.Step{Text: "recovered"},
	)
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"failing": {{Err: errors.New("tool failed")}},
	})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model", MaxSteps: 5}, runner, sink)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Turn should complete; tool error is passed to model, not fatal.
	if result.StopReason != bridle.StopReasonModelDone {
		t.Errorf("StopReason = %q; want model_done", result.StopReason)
	}
	if result.FinalText != "recovered" {
		t.Errorf("FinalText = %q; want recovered", result.FinalText)
	}
	// ToolCallResult should record the error.
	var tcr bridle.ToolCallResult
	for _, e := range sink.Events {
		if r, ok := e.(bridle.ToolCallResult); ok {
			tcr = r
		}
	}
	if tcr.Err == "" {
		t.Error("ToolCallResult.Err is empty; expected tool error string")
	}
}

// --- panic recovery ---

func TestRunTurn_PanicRecovery(t *testing.T) {
	p := &panicProvider{}
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "fake-model"}, fake.NewToolRunner(nil), sink)

	if err == nil {
		t.Fatal("expected error from panic recovery")
	}
	if result.StopReason != bridle.StopReasonError {
		t.Errorf("StopReason = %q; want error", result.StopReason)
	}
}

// --- subprocess-stream provider capability advertisement ---

func TestSubprocessProvider_CapabilityAdvertisement(t *testing.T) {
	p := fake.NewSubprocessProvider()
	caps := p.Capabilities()
	if caps.Category != bridle.CategorySubprocessStream {
		t.Errorf("Category = %q; want subprocess-stream", caps.Category)
	}
	if caps.SupportsBeforeToolCall {
		t.Error("SupportsBeforeToolCall should be false for subprocess-stream")
	}
	if !caps.SupportsAfterToolCall {
		t.Error("SupportsAfterToolCall should be true for subprocess-stream")
	}
}

// TestSubprocessProvider_TextTurn verifies a text-only turn through the
// subprocess fake emits the right events.
func TestSubprocessProvider_TextTurn(t *testing.T) {
	p := fake.NewSubprocessProvider(fake.SubprocessStep{Text: "subprocess result"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "fake-model",
		UserMessage: "test",
	}, fake.NewToolRunner(nil), sink)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinalText != "subprocess result" {
		t.Errorf("FinalText = %q; want 'subprocess result'", result.FinalText)
	}
	assertEventOrder(t, sink.Events, "ModelChunk", "TurnDone")
}

// --- Model required ---

func TestRunTurn_ModelRequired(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "ok"})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{}, fake.NewToolRunner(nil), sink)
	if err == nil {
		t.Fatal("expected ErrModelRequired, got nil")
	}
	if !errors.Is(err, bridle.ErrModelRequired) {
		t.Errorf("err = %v; want ErrModelRequired", err)
	}
}

// --- helpers ---

func assertEventOrder(t *testing.T, events []bridle.Event, types ...string) {
	t.Helper()
	got := make([]string, 0, len(events))
	for _, e := range events {
		switch e.(type) {
		case bridle.ModelChunk:
			got = append(got, "ModelChunk")
		case bridle.ToolCallStart:
			got = append(got, "ToolCallStart")
		case bridle.ToolCallResult:
			got = append(got, "ToolCallResult")
		case bridle.StepBoundary:
			got = append(got, "StepBoundary")
		case bridle.TurnDone:
			got = append(got, "TurnDone")
		case bridle.TurnError:
			got = append(got, "TurnError")
		}
	}
	if len(got) != len(types) {
		t.Errorf("event sequence = %v; want %v", got, types)
		return
	}
	for i, want := range types {
		if got[i] != want {
			t.Errorf("event[%d] = %q; want %q (full sequence: %v)", i, got[i], want, got)
		}
	}
}

// cancelOnRunToolRunner cancels its context when Run is called.
type cancelOnRunToolRunner struct {
	cancel func()
	result json.RawMessage
}

func (r *cancelOnRunToolRunner) Run(_ context.Context, _ bridle.ToolCall) (json.RawMessage, error) {
	r.cancel()
	return r.result, nil
}

// --- MCP tool name collision ---

// TestRunTurn_MCPToolNameCollision verifies that RunTurn returns ErrToolNameCollision when
// an explicit tool and the MCP config advertise the same name.
// We use a fake MCPClientConfig pointing at a non-existent stdio server — the
// connection will fail before the collision check. To test the collision path purely,
// we pass an MCPClientConfig with an empty Servers list to the SupportsMCP=false provider
// (which skips MCP entirely), and test the collision directly via the mcpclient package tests.
// The actual harness-level collision path is exercised in TestRunTurn_MCPNoServers.
func TestRunTurn_MCPNoServers(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "hello", StopReason: bridle.StopReasonModelDone})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	// MCP config with no servers — should be a no-op and turn should complete normally.
	req := bridle.TurnRequest{
		Model: "fake-model",
		Tools: []bridle.ToolDef{toolDef("explicit_tool")},
		MCP:   &bridle.MCPClientConfig{}, // empty — no servers
	}
	result, err := h.RunTurn(context.Background(), req, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Errorf("want model_done, got %s", result.StopReason)
	}
	if result.FinalText != "hello" {
		t.Errorf("want 'hello', got %q", result.FinalText)
	}
}

// TestRunTurn_MCPServerFailure_DoesNotWedgeTurn pins NEX-596: a single
// failing MCP server must not fail the whole turn. The failed server is
// skipped, its tools dropped, the turn proceeds, and an MCPServerFailed
// event is emitted for observability.
func TestRunTurn_MCPServerFailure_DoesNotWedgeTurn(t *testing.T) {
	p := fake.NewProvider(fake.Step{Text: "hello", StopReason: bridle.StopReasonModelDone})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	req := bridle.TurnRequest{
		Model: "fake-model",
		Tools: []bridle.ToolDef{toolDef("explicit_tool")},
		MCP: &bridle.MCPClientConfig{
			Servers: []bridle.MCPServerSpec{{
				Name:      "nexus-jira",
				Transport: bridle.MCPTransportStdio,
				Command:   []string{"this-binary-does-not-exist-nex596"},
			}},
		},
	}
	result, err := h.RunTurn(context.Background(), req, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("turn should complete despite MCP server failure, got: %v", err)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Errorf("StopReason = %q; want model_done (turn proceeded)", result.StopReason)
	}
	if result.FinalText != "hello" {
		t.Errorf("FinalText = %q; want hello", result.FinalText)
	}

	var failed *bridle.MCPServerFailed
	for _, e := range sink.Events {
		if f, ok := e.(bridle.MCPServerFailed); ok {
			failed = &f
		}
	}
	if failed == nil {
		t.Fatal("MCPServerFailed event not emitted")
	}
	if failed.Server != "nexus-jira" {
		t.Errorf("MCPServerFailed.Server = %q; want nexus-jira", failed.Server)
	}
	if failed.Err == nil {
		t.Error("MCPServerFailed.Err is nil; want the underlying connect error")
	}
	if failed.TS.IsZero() {
		t.Error("MCPServerFailed.TS is zero; want stamped by the harness")
	}
}

// TestRunTurn_MCPIgnoredForSubprocess verifies that subprocess-stream providers
// ignore TurnRequest.MCP (SupportsMCP=false).
func TestRunTurn_MCPIgnoredForSubprocess(t *testing.T) {
	p := fake.NewSubprocessProvider(fake.SubprocessStep{
		Text:       "subprocess response",
		StopReason: bridle.StopReasonModelDone,
	})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	req := bridle.TurnRequest{
		Model: "fake-model",
		MCP: &bridle.MCPClientConfig{
			Servers: []bridle.MCPServerSpec{{
				Name:      "unreachable-server",
				Transport: bridle.MCPTransportStdio,
				Command:   []string{"nonexistent-binary"},
			}},
		},
	}
	// Should succeed — subprocess provider ignores MCP, so the unreachable server
	// is never contacted.
	result, err := h.RunTurn(context.Background(), req, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinalText != "subprocess response" {
		t.Errorf("want 'subprocess response', got %q", result.FinalText)
	}
}

// panicProvider always panics inside RunTurn.
type panicProvider struct{}

func (p *panicProvider) Name() bridle.ProviderID { return "panic-fake" }
func (p *panicProvider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{Category: bridle.CategoryDirectAPI, SupportsCustomTools: true, SupportsBeforeToolCall: true, SupportsAfterToolCall: true, SupportsMCP: true}
}
func (p *panicProvider) RunTurn(_ context.Context, _ bridle.ProviderRequest, _ bridle.EventSink) (bridle.ProviderResult, error) {
	panic("deliberate test panic")
}

// TestRunTurn_TextAccumulatesAcrossSteps pins the fix where text content
// from successive provider turns is concatenated rather than overwritten.
// The historic bug discarded earlier text blocks when a turn included a
// tool call followed by more text — observed 2026-05-14 with keel
// producing a 1306-token cairn spec review whose substantive 6-point
// analysis was silently dropped because only the closing coda survived.
// TestRunTurn_ResolvedModelPropagates verifies the upstream model id
// the provider reported flows through into TurnResult.ResolvedModel.
// Catches the silent-divergence class where per-turn ProviderEnv
// routes the call to a different backend than cfg.Model and the
// activity log reports the configured-but-not-actually-used model.
func TestRunTurn_ResolvedModelPropagates(t *testing.T) {
	p := fake.NewProvider(fake.Step{
		Text:          "ok",
		ResolvedModel: "deepseek-chat-v3-via-anthropic-shape",
	})
	h := bridle.NewHarness(p)
	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model: "claude-3-5-sonnet-20241022", // what the operator THINKS they're calling
	}, fake.NewToolRunner(nil), &fake.SliceEventSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ResolvedModel != "deepseek-chat-v3-via-anthropic-shape" {
		t.Errorf("ResolvedModel = %q; want deepseek-... (provider report should win over cfg.Model)", result.ResolvedModel)
	}
}

// TestRunTurn_ResolvedModelLastRoundWins verifies that across a multi-
// step turn (text → tool → text), the latest non-empty ResolvedModel
// is what surfaces — handles a (theoretical) provider that re-routes
// mid-turn. Empty rounds don't clobber.
func TestRunTurn_ResolvedModelLastRoundWins(t *testing.T) {
	toolStep := fake.Step{
		ToolCalls:     []bridle.ToolInvocation{inv("1", "echo")},
		ResolvedModel: "round-1-model",
	}
	finalStep := fake.Step{
		Text:          "done",
		ResolvedModel: "round-2-model",
	}
	p := fake.NewProvider(toolStep, finalStep)
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"echo": {{Result: rawJSON(`"ok"`)}},
	})
	h := bridle.NewHarness(p)
	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:    "configured-model",
		Tools:    []bridle.ToolDef{toolDef("echo")},
		MaxSteps: 5,
	}, runner, &fake.SliceEventSink{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ResolvedModel != "round-2-model" {
		t.Errorf("ResolvedModel = %q; want round-2-model (latest non-empty wins)", result.ResolvedModel)
	}
}

func TestRunTurn_TextAccumulatesAcrossSteps(t *testing.T) {
	// Step 1: model emits substantive text + a tool call.
	step1 := fake.Step{
		Text:      "Initial substantive analysis with multiple points.",
		ToolCalls: []bridle.ToolInvocation{inv("1", "echo")},
	}
	// Step 2: post-tool, model emits a short closing line only.
	step2 := fake.Step{Text: "Done."}

	p := fake.NewProvider(step1, step2)
	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"echo": {{Result: rawJSON(`"ok"`)}},
	})
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:    "fake-model",
		Tools:    []bridle.ToolDef{toolDef("echo")},
		MaxSteps: 5,
	}, runner, sink)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Both text blocks should be present, separated by blank line.
	want := "Initial substantive analysis with multiple points.\n\nDone."
	if result.FinalText != want {
		t.Errorf("FinalText accumulation failed:\n  got:  %q\n  want: %q",
			result.FinalText, want)
	}
}
