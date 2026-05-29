# Native-API ToolRunner — P3a: bridle BeforeToolCall deny-with-result Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Extend bridle's `BeforeToolCall` hook so a hook can **deny a single tool call and supply the tool_result** (model sees it and reacts), instead of the current all-or-nothing `HookContinue`/`HookAbort`. This is the harness-layer foundation for the autonomous permission model (P3) — both deny-as-tool-error and (later) chat-based operator escalation route through it.

**Architecture:** Carry the deny verdict in the already-mutable `BeforeToolCallCtx` struct (passed in and returned by the hook). A hook sets `Deny=true` (+ `Result`/`Err`) and returns `HookContinue`. `executeToolCall` checks `btc.Deny` after running the BeforeToolCall hooks: if set, skip `runner.Run`/MCP, build the `tool_result` from the hook's `Result`/`Err`, still fire `AfterToolCall`, and continue the loop (NOT abort). No `HookAction` enum change, no `runHooks` change.

**Tech Stack:** Go, bridle (`hooks.go`, `run.go`), existing `fake` provider + `fake.ToolRunner` for tests.

**Why this design (decision recorded):** operator chose "extend bridle's hook" over a wrapping decorator — enforcement lives at the harness layer. bridle's hook contract today only supports abort-the-whole-turn; this adds the missing "deny one call, hand the model an error it can react to" path. Design ref: native-api-toolrunner-design.md §8.

**Scope / non-goals:** bridle change ONLY. The per-aspect policy + the funnel hook that USES this (allow/deny) is P3b (nexus, needs this merged + a bridle bump — same pattern as P2). Chat escalation + operator-attention mechanism is P3c (its own design; the funnel hook will block on a chat round-trip then set Deny or not). None of that is built here.

---

## Pre-flight reads (confirm before editing)

- `hooks.go:45-49` — `BeforeToolCallCtx` (add fields here); confirm the file's imports (needs `encoding/json`).
- `run.go:238-300` — `executeToolCall`: the exact construction of `tcr := ToolCallResult{ID, Result, Err}`, the `sink.Emit(tcr)`, the `AfterToolCall` block, and how `tcr` becomes the returned `toolMsg ProviderMessage` (Role `"tool_result"`) + the `completed ToolInvocation`. The deny branch must mirror this exactly.
- Confirm `ToolCallResult` and `ProviderMessage` field names used in the normal path.

---

## Task 1: Add deny fields to BeforeToolCallCtx (TDD)

**Files:** Modify `hooks.go`. Test in `hooks_deny_test.go` (or extend an existing run-loop test file — check for `run_internal_test.go`).

- [ ] **Step 1: Add fields** to `BeforeToolCallCtx` in `hooks.go` (add `import "encoding/json"` if absent):

```go
// BeforeToolCallCtx carries context passed to BeforeToolCall hooks.
type BeforeToolCallCtx struct {
	Call ToolCall
	Step int

	// Deny, when set by a BeforeToolCall hook, tells the harness to SKIP
	// executing this tool call and instead return Result/Err as the
	// tool_result, then continue the loop so the model can react. Use this
	// (with a returned HookContinue) for permission denials — HookAbort
	// ends the whole turn, which is not what a per-call denial wants.
	Deny bool
	// Result is the tool_result JSON payload when Deny is set. nil → null.
	Result json.RawMessage
	// Err, when non-empty and Deny is set, marks the tool_result as an
	// error string the model sees (mirrors a runner.Run error).
	Err string
}
```

- [ ] **Step 2: Failing test** — a BeforeToolCall hook denies the "bash" tool; assert the runner is NOT called for it, the model receives the deny result, and the turn does NOT abort. Use the fake provider scripted to call `bash` once then finish, and a `fake.ToolRunner` that records calls. Skeleton (adapt to the real fake provider's scripting API — read an existing run-loop test first):

```go
func TestBeforeToolCallDenySkipsExecutionAndContinues(t *testing.T) {
	// fake provider: step 0 → emits a tool_use for "bash"; step 1 → final text.
	prov := /* fake provider scripted: [toolUse("bash", `{"command":"rm -rf /"}`), finalText("ok, I won't")] */
	h := bridle.NewHarness(prov)

	denied := false
	h.RegisterBeforeToolCall(func(ctx context.Context, in bridle.BeforeToolCallCtx) (bridle.BeforeToolCallCtx, bridle.HookAction, error) {
		if in.Call.Name == "bash" {
			in.Deny = true
			in.Err = "permission denied by policy: bash"
			denied = true
		}
		return in, bridle.HookContinue, nil
	})

	runner := fake.NewToolRunner(map[string][]fake.ToolResult{
		"bash": {{Result: json.RawMessage(`{"stdout":"SHOULD NOT RUN"}`)}},
	})
	res, err := h.RunTurn(ctx, req /* Tools: bash def; MaxSteps:4 */, runner, &fake.SliceEventSink{})

	if err != nil { t.Fatal(err) }
	if !denied { t.Fatal("hook did not see the bash call") }
	if res.StopReason == bridle.StopReasonAborted { t.Fatal("deny must NOT abort the turn") }
	// the recorded ToolInvocation for bash should carry the deny error, not the runner's output:
	var bash *bridle.ToolInvocation
	for i := range res.ToolCalls { if res.ToolCalls[i].Name == "bash" { bash = &res.ToolCalls[i] } }
	if bash == nil || bash.Err == "" || strings.Contains(string(bash.Result), "SHOULD NOT RUN") {
		t.Fatalf("bash should be denied (err set, runner not invoked), got %+v", bash)
	}
}
```

- [ ] **Step 3: Run, expect FAIL** — `go test . -run TestBeforeToolCallDeny -v` (deny is ignored today → runner runs → SHOULD NOT RUN appears).

---

## Task 2: Honour Deny in executeToolCall (TDD → green)

**Files:** Modify `run.go` (`executeToolCall`).

- [ ] **Step 1:** In `executeToolCall`, immediately after `call = btc.Call` (and the existing `err || aborted` early return), add the deny branch BEFORE the `ctx.Err()` / execution block:

```go
	call = btc.Call

	if btc.Deny {
		resultJSON := btc.Result
		if resultJSON == nil {
			resultJSON = json.RawMessage(`null`)
		}
		tcr := ToolCallResult{ID: call.ID, Result: resultJSON, Err: btc.Err}
		sink.Emit(tcr)
		// AfterToolCall still fires so audit/observability sees the denial.
		atc := AfterToolCallCtx{Call: call, Result: tcr, Step: stepCount + 1}
		atc, aborted, err = h.hooks.runAfterToolCall(ctx, atc)
		if err != nil || aborted {
			return
		}
		tcr = atc.Result
		// Build the same tool_result message + completed invocation the
		// normal path produces (mirror the construction below).
		toolMsg = /* ProviderMessage{Role:"tool_result", ...} exactly as normal path builds from tcr */
		completed = ToolInvocation{ID: call.ID, Name: call.Name, Args: call.Args, Result: tcr.Result, Err: tcr.Err}
		return toolMsg, completed, false, nil
	}
```

> Read the normal path's `tcr → toolMsg` construction (run.go ~285-300) and replicate it exactly in the `toolMsg = ...` line — same fields, same content blocks. Do not invent a new shape.

- [ ] **Step 2: Run, expect PASS** — `go test . -run TestBeforeToolCallDeny -v`.
- [ ] **Step 3: Full bridle suite + vet** — `go test ./... && go vet ./...` (confirm no regression in existing hook/tool-loop tests: `run_session_tool_replay_test.go` etc.).
- [ ] **Step 4: Commit** — `git commit -am "feat(hooks): BeforeToolCall can deny a single call with a tool_result (continue, not abort)"`

---

## Task 3: Doc + example (the permission-deny pattern)

- [ ] **Step 1:** Add a short doc comment block (in `hooks.go` near `BeforeToolCallCtx`, or a `doc.go`) showing the canonical permission-deny hook: check policy → `in.Deny=true; in.Err="denied: <reason>"` → return `HookContinue`. Note that returning `HookAbort` is for killing the turn, `Deny` is for refusing one call.
- [ ] **Step 2: Commit** — `git commit -am "docs(hooks): document the BeforeToolCall permission-deny pattern"`

---

## Self-Review

- **Behaviour:** deny skips execution, returns the supplied result/err as the tool_result, continues the loop (not abort) ✓. Hooks can still abort (HookAbort) and still mutate the call (`btc.Call`) ✓ — additive, no behaviour change for existing hooks (Deny defaults false).
- **Audit:** AfterToolCall fires on a denied call so observability sees it ✓.
- **Escalation-ready:** a funnel hook can block (sync + ctx) doing a chat round-trip, then set Deny or leave it — same mechanism (P3c) ✓.
- **Scope:** bridle-only; no policy, no nexus, no escalation here ✓.
- **Type consistency:** new fields `Deny`/`Result`/`Err`; deny branch mirrors the normal-path `tcr→toolMsg` construction (verify field names against run.go) ✓.

---

## Follow-on phases (outline, NOT built here)

- **P3b (nexus):** bump bridle pin to the merged P3a commit; add a per-aspect **policy** (per-tool allow/deny + bash-command / write-path arg patterns) and a funnel `BeforeToolCall` hook that consults it → allow (Continue) / deny (`Deny=true`, Err=reason). Default postures: local bash/write guarded; host comms light; (external MCP = P4). Prove with a live DeepSeek chain: aspect attempts a denied `bash` → gets the refusal → adapts. Same merge-then-bump dance as P2.
- **P3c (escalation):** on an "escalate" verdict the funnel hook posts an approval request to **chat** and triggers an **operator-attention mechanism** (TBD — explore nexus's notification/agora-alert options; candidates: a flagged/mention chat frame, an agora alert, a push), then **blocks** awaiting the operator's chat decision (with a timeout → default-deny), then sets `Deny` or allows. Needs its own design pass (operator-side approve/deny UX + the attention channel + timeout policy).
