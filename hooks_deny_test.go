package bridle_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

// recordingToolRunner wraps a fake.ToolRunner and records the names of
// every tool the harness actually dispatched to, so a test can assert a
// denied call was never executed.
type recordingToolRunner struct {
	inner *fake.ToolRunner
	calls []string
}

func (r *recordingToolRunner) Run(ctx context.Context, call bridle.ToolCall) (json.RawMessage, error) {
	r.calls = append(r.calls, call.Name)
	return r.inner.Run(ctx, call)
}

// A BeforeToolCall hook that sets Deny=true (with an Err) and returns
// HookContinue must cause the harness to SKIP the runner for that call,
// hand the model the deny Err as the tool_result, and CONTINUE the loop
// (not abort). The model then emits its final text and the turn ends
// normally.
func TestBeforeToolCallDenySkipsExecutionAndContinues(t *testing.T) {
	// Step 0: model emits a tool_use for "bash". Step 1: final text.
	p := fake.NewProvider(
		fake.Step{ToolCalls: []bridle.ToolInvocation{
			{ID: "1", Name: "bash", Args: json.RawMessage(`{"command":"rm -rf /"}`)},
		}},
		fake.Step{Text: "ok, I won't"},
	)
	h := bridle.NewHarness(p)

	denied := false
	h.RegisterBeforeToolCall(func(ctx context.Context, in bridle.BeforeToolCallCtx) (bridle.BeforeToolCallCtx, bridle.HookAction, error) {
		if in.Call.Name == "bash" {
			in.Deny = true
			in.Err = "permission denied by policy: bash"
			denied = true
		}
		return in, bridle.HookContinue, nil
	})

	// Runner is scripted with a result that must NOT surface — if the
	// deny is honoured the runner is never invoked for bash.
	runner := &recordingToolRunner{inner: fake.NewToolRunner(map[string][]fake.ToolResult{
		"bash": {{Result: json.RawMessage(`{"stdout":"SHOULD NOT RUN"}`)}},
	})}

	res, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:    "fake-model",
		Tools:    []bridle.ToolDef{{Name: "bash", Description: "bash", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		MaxSteps: 4,
	}, runner, &fake.SliceEventSink{})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !denied {
		t.Fatal("hook did not see the bash call")
	}
	if res.StopReason == bridle.StopReasonAborted {
		t.Fatalf("deny must NOT abort the turn; StopReason=%q", res.StopReason)
	}

	// The runner must never have been invoked for the denied call.
	for _, name := range runner.calls {
		if name == "bash" {
			t.Errorf("runner was invoked for denied tool %q; deny did not skip execution (calls=%v)", name, runner.calls)
		}
	}

	// The recorded ToolInvocation for bash should carry the deny error,
	// not the runner's output.
	var bash *bridle.ToolInvocation
	for i := range res.ToolCalls {
		if res.ToolCalls[i].Name == "bash" {
			bash = &res.ToolCalls[i]
		}
	}
	if bash == nil {
		t.Fatalf("no bash ToolInvocation recorded; got %+v", res.ToolCalls)
	}
	if bash.Err != "permission denied by policy: bash" {
		t.Errorf("bash.Err = %q; want the deny error", bash.Err)
	}
	if strings.Contains(string(bash.Result), "SHOULD NOT RUN") {
		t.Errorf("bash result carries runner output %q; runner should not have run", string(bash.Result))
	}

	// The turn should have continued to its final text.
	if !strings.Contains(res.FinalText, "ok, I won't") {
		t.Errorf("FinalText = %q; want the post-deny model text", res.FinalText)
	}
}
