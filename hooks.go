package bridle

import (
	"context"
	"encoding/json"
)

// HookAction tells the harness what to do after a hook returns.
type HookAction int

const (
	HookContinue HookAction = iota
	HookAbort               // end the turn; partial TurnResult returned with StopReason=aborted
)

// Hook is the generic hook signature. T is the mutable context value passed
// in and returned. Registration order is the execution order.
type Hook[T any] func(ctx context.Context, in T) (T, HookAction, error)

// HookID identifies a registered hook so it can be removed via
// Harness.UnregisterHook. The zero value is not a valid hook id —
// every successful Register* call returns a non-zero id.
//
// Hook registration is NOT safe to call concurrently with RunTurn or
// with itself; mirror the existing assumption that hooks are wired
// during setup. If you need to swap a hook between turns, do it from
// a single goroutine while no turn is in flight.
type HookID uint64

// BeforeModelCallCtx carries context passed to BeforeModelCall hooks.
//
// Request points at the live ProviderRequest that the harness is about
// to send to the provider — the same struct, not a copy. Hooks may
// mutate its fields in place (Model, AppendSystemPrompt, Tools,
// ProviderEnv, Messages, etc.) and the changes apply to the upcoming
// call. The hook fires once before the initial call (Step=0) and once
// before every subsequent call inside the tool loop (Step=N), so
// per-step mutations (escalate model on N, drop a tool once used) work.
//
// Mutating Messages is supported but advanced: by the time the in-loop
// hook fires, the harness has already appended the assistant tool_use
// turn and the tool_result blocks for the round just completed.
type BeforeModelCallCtx struct {
	Request *ProviderRequest
	Step    int
}

// BeforeToolCallCtx carries context passed to BeforeToolCall hooks.
//
// Permission-deny pattern: a BeforeToolCall hook can refuse a single
// tool call without ending the turn by setting Deny=true (optionally
// with Result and/or Err) and returning HookContinue. The harness then
// SKIPS executing that call (no runner.Run, no MCP dispatch), builds the
// tool_result from Result/Err, fires AfterToolCall for audit, and
// continues the loop so the model sees the refusal and can react.
// Contrast HookAbort, which ends the whole turn — use Deny for refusing
// one call, HookAbort for killing the turn.
//
// Canonical hook:
//
//	h.RegisterBeforeToolCall(func(ctx context.Context, in bridle.BeforeToolCallCtx) (bridle.BeforeToolCallCtx, bridle.HookAction, error) {
//		if !policy.Allows(in.Call) {
//			in.Deny = true
//			in.Err = "permission denied by policy: " + in.Call.Name
//		}
//		return in, bridle.HookContinue, nil
//	})
type BeforeToolCallCtx struct {
	Call ToolCall
	Step int

	// Deny, when set true by a BeforeToolCall hook, tells the harness to
	// SKIP executing this tool call and instead return Result/Err as the
	// tool_result, then continue the loop so the model can react. Use this
	// (with a returned HookContinue) for per-call permission denials —
	// HookAbort ends the whole turn, which is not what a single-call
	// denial wants. Defaults false: existing hooks are unaffected.
	Deny bool
	// Result is the tool_result JSON payload to hand back when Deny is
	// set. nil is treated as JSON null. Ignored unless Deny is true.
	Result json.RawMessage
	// Err, when non-empty and Deny is set, marks the tool_result as an
	// error string the model sees (mirrors a runner.Run error). Ignored
	// unless Deny is true.
	Err string
}

// AfterToolCallCtx carries context passed to AfterToolCall hooks.
type AfterToolCallCtx struct {
	Call   ToolCall
	Result ToolCallResult
	Step   int
}

// OnStepBoundaryCtx carries context passed to OnStepBoundary hooks.
type OnStepBoundaryCtx struct {
	Step int
}

// OnTurnDoneCtx carries context passed to OnTurnDone hooks.
// Hooks may mutate SessionDelta before it is returned to the funnel.
type OnTurnDoneCtx struct {
	Result *TurnResult
}

// hookEntry binds a registered hook to its id so UnregisterHook can
// find and remove it.
type hookEntry[T any] struct {
	id HookID
	fn Hook[T]
}

// hookRegistry holds all registered hooks for a Harness instance.
type hookRegistry struct {
	nextID          HookID
	beforeModelCall []hookEntry[BeforeModelCallCtx]
	beforeToolCall  []hookEntry[BeforeToolCallCtx]
	afterToolCall   []hookEntry[AfterToolCallCtx]
	onStepBoundary  []hookEntry[OnStepBoundaryCtx]
	onTurnDone      []hookEntry[OnTurnDoneCtx]
}

func (r *hookRegistry) newID() HookID {
	r.nextID++
	return r.nextID
}

// RegisterBeforeModelCall adds a hook that fires before each model
// invocation. Returns a HookID that can be passed to UnregisterHook.
func (h *Harness) RegisterBeforeModelCall(fn Hook[BeforeModelCallCtx]) HookID {
	id := h.hooks.newID()
	h.hooks.beforeModelCall = append(h.hooks.beforeModelCall, hookEntry[BeforeModelCallCtx]{id, fn})
	return id
}

// RegisterBeforeToolCall adds a hook that fires before each tool
// execution. Returns a HookID that can be passed to UnregisterHook.
//
// To refuse a single call without ending the turn, set in.Deny=true
// (with in.Err / in.Result) and return HookContinue — see
// BeforeToolCallCtx for the permission-deny pattern. Return HookAbort
// only to kill the whole turn.
func (h *Harness) RegisterBeforeToolCall(fn Hook[BeforeToolCallCtx]) HookID {
	id := h.hooks.newID()
	h.hooks.beforeToolCall = append(h.hooks.beforeToolCall, hookEntry[BeforeToolCallCtx]{id, fn})
	return id
}

// RegisterAfterToolCall adds a hook that fires after each tool
// execution. Returns a HookID that can be passed to UnregisterHook.
func (h *Harness) RegisterAfterToolCall(fn Hook[AfterToolCallCtx]) HookID {
	id := h.hooks.newID()
	h.hooks.afterToolCall = append(h.hooks.afterToolCall, hookEntry[AfterToolCallCtx]{id, fn})
	return id
}

// RegisterOnStepBoundary adds a hook that fires between tool-call
// rounds. Returns a HookID that can be passed to UnregisterHook.
func (h *Harness) RegisterOnStepBoundary(fn Hook[OnStepBoundaryCtx]) HookID {
	id := h.hooks.newID()
	h.hooks.onStepBoundary = append(h.hooks.onStepBoundary, hookEntry[OnStepBoundaryCtx]{id, fn})
	return id
}

// RegisterOnTurnDone adds a hook that fires after the turn completes.
// Hooks may mutate TurnResult.SessionDelta. Returns a HookID that can
// be passed to UnregisterHook.
func (h *Harness) RegisterOnTurnDone(fn Hook[OnTurnDoneCtx]) HookID {
	id := h.hooks.newID()
	h.hooks.onTurnDone = append(h.hooks.onTurnDone, hookEntry[OnTurnDoneCtx]{id, fn})
	return id
}

// UnregisterHook removes the hook with the given id from whichever
// hook slice it was registered into. Returns true if a hook was
// removed, false if no hook with that id exists. The zero HookID is
// never registered and always returns false.
//
// Not safe to call concurrently with RunTurn or with Register*. See
// HookID for the threading contract.
func (h *Harness) UnregisterHook(id HookID) bool {
	if id == 0 {
		return false
	}
	r := &h.hooks
	if next, ok := removeHookByID(r.beforeModelCall, id); ok {
		r.beforeModelCall = next
		return true
	}
	if next, ok := removeHookByID(r.beforeToolCall, id); ok {
		r.beforeToolCall = next
		return true
	}
	if next, ok := removeHookByID(r.afterToolCall, id); ok {
		r.afterToolCall = next
		return true
	}
	if next, ok := removeHookByID(r.onStepBoundary, id); ok {
		r.onStepBoundary = next
		return true
	}
	if next, ok := removeHookByID(r.onTurnDone, id); ok {
		r.onTurnDone = next
		return true
	}
	return false
}

// removeHookByID returns a new slice with the entry matching id
// removed, along with a flag indicating whether removal happened. The
// original slice is not mutated.
func removeHookByID[T any](entries []hookEntry[T], id HookID) ([]hookEntry[T], bool) {
	for i, e := range entries {
		if e.id == id {
			out := make([]hookEntry[T], 0, len(entries)-1)
			out = append(out, entries[:i]...)
			out = append(out, entries[i+1:]...)
			return out, true
		}
	}
	return entries, false
}

// runBeforeModelCall fires all BeforeModelCall hooks in registration order.
// Returns (updated ctx, aborted, error).
func (r *hookRegistry) runBeforeModelCall(ctx context.Context, hc BeforeModelCallCtx) (BeforeModelCallCtx, bool, error) {
	return runHooks(ctx, hc, r.beforeModelCall)
}

func (r *hookRegistry) runBeforeToolCall(ctx context.Context, hc BeforeToolCallCtx) (BeforeToolCallCtx, bool, error) {
	return runHooks(ctx, hc, r.beforeToolCall)
}

func (r *hookRegistry) runAfterToolCall(ctx context.Context, hc AfterToolCallCtx) (AfterToolCallCtx, bool, error) {
	return runHooks(ctx, hc, r.afterToolCall)
}

func (r *hookRegistry) runOnStepBoundary(ctx context.Context, hc OnStepBoundaryCtx) (OnStepBoundaryCtx, bool, error) {
	return runHooks(ctx, hc, r.onStepBoundary)
}

func (r *hookRegistry) runOnTurnDone(ctx context.Context, hc OnTurnDoneCtx) (OnTurnDoneCtx, bool, error) {
	return runHooks(ctx, hc, r.onTurnDone)
}

func runHooks[T any](ctx context.Context, in T, hooks []hookEntry[T]) (T, bool, error) {
	cur := in
	for _, h := range hooks {
		out, action, err := h.fn(ctx, cur)
		if err != nil {
			return cur, false, err
		}
		cur = out
		if action == HookAbort {
			return cur, true, nil
		}
	}
	return cur, false, nil
}
