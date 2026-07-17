package bridle

import (
	"context"
	"encoding/json"
)

// ToolDef describes a tool the model may call.
type ToolDef struct {
	Name        string
	Description string
	// InputSchema is a JSON Schema object describing the expected arguments.
	InputSchema json.RawMessage
}

// ToolCall is a single invocation the model requested.
type ToolCall struct {
	ID   string
	Name string
	Args json.RawMessage
}

// ToolRunner executes tool calls on behalf of the harness.
// The funnel supplies the implementation; the harness never owns tools.
type ToolRunner interface {
	Run(ctx context.Context, call ToolCall) (json.RawMessage, error)
}

// ToolResult is what a ToolExecutor returns for one tool call: the
// tool_result payload, or an error string if the runner failed. Mirrors
// the shape ToolCallResult carries onto the event sink, minus the id/TS
// bookkeeping fields the harness owns.
type ToolResult struct {
	Result json.RawMessage
	Err    string
}

// ToolExecutor lets a subprocess-stream provider whose agentic loop runs
// mid-RunTurn (bridle spec NEX-745 §4 — claudesdk today) service a
// bridle-defined tool call the SAME way direct-api providers do: through
// the harness's own ToolRunner-backed pipeline, so OnBeforeToolCall /
// OnAfterToolCall fire with identical semantics and the call is visible
// on the event sink exactly like any other tool invocation.
//
// The harness supplies this via ProviderRequest.ToolExecutor; it is nil
// for every provider that doesn't ask for it (every provider except
// claudesdk today — direct-api providers own their own tool loop in
// run.go and never read this field, subprocess-stream providers that
// don't support custom tools, e.g. claudecode, don't get one either).
//
// Invariant (asserted by callers, not by this interface): every ToolCall
// id passed to Execute gets exactly one ToolResult or error before the
// provider's RunTurn returns.
type ToolExecutor interface {
	Execute(ctx context.Context, call ToolCall) (ToolResult, error)
}
