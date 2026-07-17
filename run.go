package bridle

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/CarriedWorldUniverse/bridle/internal/mcpclient"
)

// runTurn is the inner implementation, called by RunTurn after the panic trap.
func (h *Harness) runTurn(ctx context.Context, req TurnRequest, runner ToolRunner, sink EventSink) (TurnResult, error) {
	now := h.clock()
	turnStart := now()
	var timing TurnTiming

	// Wrap the caller's sink ONCE: every event the harness or the
	// provider emits from here on is timestamp-stamped, and the
	// decorator records each round's first-event time for the
	// startup/stream split. Providers receive the wrapped sink, so
	// provider-emitted events get stamped for free — zero provider
	// changes.
	ssink := &stampSink{inner: sink, now: now}
	sink = ssink

	// Connect MCP servers and merge tool surface (direct-api providers only).
	var mcpClient *mcpclient.Client
	caps := h.provider.Capabilities()
	if caps.SupportsMCP && req.MCP != nil {
		specs := lowerMCPConfig(req.MCP)
		var err error
		mcpClient, err = mcpclient.Connect(ctx, specs)
		if err != nil {
			// NEX-596: per-server failures no longer hard-fail Connect.
			// A genuine returned error here is catastrophic — preserve
			// the StopReasonError path for it.
			return TurnResult{StopReason: StopReasonError}, err
		}
		defer mcpClient.Close()

		// NEX-596: a failing MCP server is skipped, not fatal. Surface
		// each dropped server for observability and proceed with the
		// tools from the servers that connected.
		for _, f := range mcpClient.Failures() {
			sink.Emit(MCPServerFailed{Server: f.Name, Err: f.Err})
		}

		mcpTools := mcpClient.Tools()
		merged, err := mergeToolSurface(req.Tools, mcpTools)
		if err != nil {
			return TurnResult{StopReason: StopReasonError}, err
		}
		req.Tools = merged
	}

	// Assembly span, round 0: lowerRequest + BeforeModelCall hooks.
	assemblyStart := now()

	// Lower TurnRequest → ProviderRequest.
	preq := lowerRequest(req)

	// NEX-745 §4: wire a ToolExecutor for subprocess-stream providers
	// whose agentic loop runs mid-RunTurn (claudesdk) — the sidecar
	// reports a custom tool call from INSIDE the provider's single
	// RunTurn invocation and must service it there, not wait for
	// run.go's per-round tool loop below. The adapter delegates to
	// h.executeToolCall, the exact path direct-api providers use, so
	// hooks/sink events fire identically. Every other provider gets a
	// nil ToolExecutor and is unaffected: direct-api providers own their
	// tool loop in this file and never read the field; subprocess-
	// stream providers that don't support custom tools (claudecode)
	// don't request one either.
	if caps.Category == CategorySubprocessStream && caps.SupportsCustomTools {
		preq.ToolExecutor = &toolExecutorAdapter{h: h, runner: runner, mcpClient: mcpClient, sink: sink}
	}

	// BeforeModelCall hook (step 0 = the initial call). Hook receives
	// &preq so mutations to Model/Tools/Messages/etc. apply to the
	// upcoming call.
	hc := BeforeModelCallCtx{Request: &preq, Step: 0}
	var aborted bool
	var herr error
	_, aborted, herr = h.hooks.runBeforeModelCall(ctx, hc)
	if herr != nil {
		return partialAbort(), herr
	}
	if aborted {
		return partialAbort(), nil
	}
	assemblySecs := now().Sub(assemblyStart).Seconds()

	var (
		allInvocations []ToolInvocation
		totalUsage     Usage
		stepCount      int
		finalText      string
		stopReason     StopReason
		sessionDelta   []SessionEvent
		resolvedModel  string // latest non-empty ResolvedModel from any round
	)

	for {
		if ctx.Err() != nil {
			return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), nil
		}

		// One RoundTiming entry per provider call. Request-size fields
		// reflect the request as the provider receives it (post-hook).
		timing.Rounds = append(timing.Rounds, RoundTiming{
			AssemblySecs: assemblySecs,
			PromptBytes:  promptBytes(preq.Messages),
			MessageCount: len(preq.Messages),
			ToolDefCount: len(preq.Tools),
		})

		// Context contract (NEX-581), PromptBudget lever: engine-agnostic
		// budget check on the assembled prompt, in TOKEN terms (reusing
		// the usage contract's estimator). v1 is warn-only — emit a
		// ContextBudgetWarning when the estimate meets/exceeds the policy
		// budget; never truncate or hard-fail. Zero PromptBudget = no check.
		if req.ContextPolicy.PromptBudget > 0 {
			assembled := estimateTokens(promptText(preq))
			if assembled >= req.ContextPolicy.PromptBudget {
				sink.Emit(ContextBudgetWarning{
					Assembled: assembled,
					Budget:    req.ContextPolicy.PromptBudget,
				})
			}
		}

		// Run the provider turn. The timing of this round lands in the
		// RoundTiming entry appended above (round = last entry).
		presult, err := h.runProviderRound(ctx, preq, sink, ssink, now, &timing.Rounds[len(timing.Rounds)-1])
		if err != nil {
			timing.TotalSecs = now().Sub(turnStart).Seconds()
			sink.Emit(TurnError{Err: err, Stage: TurnErrorStageProvider})
			return TurnResult{
				FinalText:  finalText,
				ToolCalls:  allInvocations,
				StepCount:  stepCount,
				Usage:      totalUsage,
				StopReason: StopReasonError,
				Timing:     timing,
			}, err
		}

		// NEX-581 tool-call contract: after the provider returns, before
		// tool execution, guarantee a clean tool call or clean text turn.
		// On a detected leak this repairs structurally and — under the
		// repair-then-retry strictness — retries the round ONCE. A retry
		// REPLACES the bad round: the retried presult overwrites this
		// round's timing entry and we count only the retried usage, never
		// both (the bad round is discarded, not summed). See enforceToolCallContract.
		presult, err = h.enforceToolCallContract(ctx, preq, presult, sink, ssink, now, &timing.Rounds[len(timing.Rounds)-1])
		if err != nil {
			timing.TotalSecs = now().Sub(turnStart).Seconds()
			sink.Emit(TurnError{Err: err, Stage: TurnErrorStageProvider})
			return TurnResult{
				FinalText:  finalText,
				ToolCalls:  allInvocations,
				StepCount:  stepCount,
				Usage:      totalUsage,
				StopReason: StopReasonError,
				Timing:     timing,
			}, err
		}

		// Concatenate text across provider turns rather than overwriting.
		// Multi-step turns (text → tool → text → tool → final text) need
		// every text block preserved or downstream consumers (e.g. nexus
		// funnel auto-post) see only the last fragment. Surfaced
		// 2026-05-14: keel's 6-point cairn spec review (1306 output
		// tokens across multiple text blocks) reached chat as only its
		// 232-char closing coda — the substantive analysis was discarded
		// by the overwrite. Blank-line separator preserves block
		// boundaries readers expect.
		if presult.FinalText != "" {
			if finalText != "" {
				finalText += "\n\n"
			}
			finalText += presult.FinalText
		}
		// Usage contract (NEX-581): normalize the round's usage before
		// folding it into the turn total. A provider that reported real
		// counts passes through untouched; one that returned zero gets a
		// flagged, tokenizer-estimated floor so the turn never reports
		// silently-zero usage. Engine-agnostic — same gate for every
		// provider. The estimate is sourced from this round's prompt
		// (the assembled request) and response (FinalText).
		roundUsage := normalizeUsage(presult.Usage, promptText(preq), presult.FinalText)
		totalUsage = addUsage(totalUsage, roundUsage)
		sessionDelta = append(sessionDelta, presult.SessionDelta...)
		// Track the most recent non-empty ResolvedModel — last round
		// wins, so multi-step turns where the model id might shift
		// (theoretical: a future provider that re-routes mid-turn)
		// report the final upstream identity. In practice all current
		// providers report the same model_id every round of a turn.
		if presult.ResolvedModel != "" {
			resolvedModel = presult.ResolvedModel
		}

		// Self-executing providers (claude-code, gemini-cli — anything
		// whose subprocess runs the tools itself) populate ToolCalls as
		// a record of work already done, not a request to bridle. Don't
		// re-execute via runner.Run; don't re-invoke the provider with
		// synthesized tool_results. NEX-251: re-invoking caused the
		// claudecode buildPrompt path to refire the original user prompt
		// as -p on a second `claude -p --resume` call, doubling the
		// session jsonl entries the model sees on subsequent turns.
		// Record the invocations for observability and exit the loop.
		if !caps.SupportsCustomTools {
			allInvocations = append(allInvocations, presult.ToolCalls...)
			stopReason = presult.StopReason
			break
		}

		// NEX-745: claudesdk sets SupportsCustomTools=true (it's the
		// point of the lane) but its whole tool round-trip already
		// happened INSIDE the provider.RunTurn call above, serviced via
		// ProviderRequest.ToolExecutor (the toolExecutorAdapter wired in
		// above — same executeToolCall path, so hooks/sink events
		// already fired). Its ProviderResult.ToolCalls therefore comes
		// back empty; this branch's "turn is done" path is what a
		// completed claudesdk turn hits — never the tool-execution loop
		// below, which would otherwise re-run the same calls a second
		// time via runner.Run.
		//
		// No tool calls → turn is done.
		if len(presult.ToolCalls) == 0 {
			stopReason = presult.StopReason
			break
		}

		// Execute each tool call.
		var toolMessages []ProviderMessage
		for _, inv := range presult.ToolCalls {
			toolStart := now()
			toolMsg, completed, abortedExec, execErr := h.executeToolCall(ctx, inv, stepCount, runner, mcpClient, sink)
			toolSecs := now().Sub(toolStart).Seconds()
			if execErr != nil {
				return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), execErr
			}
			if abortedExec {
				return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), nil
			}
			timing.Tools = append(timing.Tools, ToolTiming{
				ID:   completed.ID,
				Name: completed.Name,
				Secs: toolSecs,
			})
			allInvocations = append(allInvocations, completed)
			toolMessages = append(toolMessages, toolMsg)
			sessionDelta = append(sessionDelta, SessionEvent{
				Provider:   h.provider.Name(),
				Role:       RoleTool,
				Content:    toolMsg.Content,
				ToolCallID: completed.ID,
			})
		}

		stepCount++

		// OnStepBoundary hook.
		sbc := OnStepBoundaryCtx{Step: stepCount}
		_, aborted, herr = h.hooks.runOnStepBoundary(ctx, sbc)
		if herr != nil {
			return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), herr
		}
		if aborted {
			return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), nil
		}
		sink.Emit(StepBoundary{Step: stepCount})

		// MaxSteps guard.
		if req.MaxSteps > 0 && stepCount >= req.MaxSteps {
			stopReason = StopReasonMaxSteps
			break
		}

		// Assembly span, round N: rebuild of the next request (message
		// appends) + the in-loop BeforeModelCall hooks.
		assemblyStart = now()

		// Reconstruct the assistant turn that emitted those tool_use blocks
		// before appending the tool_results. Bedrock (and strict providers)
		// require assistant{tool_use} → user{tool_result} alternation; sending
		// tool_results without the preceding assistant turn is rejected.
		// finalText may be empty for tool-only assistant turns — that's fine,
		// providers emit a content-less assistant message with just tool_use.
		//
		// NEX-320: thread ThinkingBlocks through so claude's
		// toClaudeMessages can re-emit them in the next API call.
		// Anthropic API rejects multi-turn requests whose history is
		// missing the thinking blocks from prior assistant turns.
		// Other providers ignore the field.
		//
		// NEX-340: same shape on the openai side — DeepSeek's reasoner
		// models require reasoning_content to round-trip on the in-turn
		// follow-up call after a tool_call. Without this, step 2 hits
		// "The reasoning_content in the thinking mode must be passed
		// back to the API" 400.
		preq.Messages = append(preq.Messages, ProviderMessage{
			Role:             "assistant",
			Content:          finalText,
			ToolCalls:        presult.ToolCalls,
			ThinkingBlocks:   presult.ThinkingBlocks,
			ReasoningContent: presult.ReasoningContent,
		})

		// Append tool results to message history.
		preq.Messages = append(preq.Messages, toolMessages...)

		// BeforeModelCall hook for the next round. Hook receives &preq so
		// per-step mutations (escalate model, drop a tool, append a
		// system reminder message) apply to the next provider call.
		hc = BeforeModelCallCtx{Request: &preq, Step: stepCount}
		_, aborted, herr = h.hooks.runBeforeModelCall(ctx, hc)
		if herr != nil {
			return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), herr
		}
		if aborted {
			return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), nil
		}
		assemblySecs = now().Sub(assemblyStart).Seconds()
	}

	timing.TotalSecs = now().Sub(turnStart).Seconds()
	result := TurnResult{
		FinalText:     finalText,
		ToolCalls:     allInvocations,
		StepCount:     stepCount,
		Usage:         totalUsage,
		StopReason:    stopReason,
		ResolvedModel: resolvedModel,
		SessionDelta:  sessionDelta,
		Timing:        timing,
	}

	// OnTurnDone hook — may mutate SessionDelta.
	otd := OnTurnDoneCtx{Result: &result}
	h.hooks.runOnTurnDone(ctx, otd) //nolint:errcheck
	sink.Emit(TurnDone{Result: result})
	return result, nil
}

// runProviderRound runs one provider turn and records its timing into the
// supplied RoundTiming entry, populating the startup/stream split from the
// stampSink's first-event mark. Factored out of the round loop so the
// tool-call contract's retry (NEX-581) can re-run the provider and OVERWRITE
// the same RoundTiming entry — the retry replaces the bad round, so its
// timing supersedes rather than adds to the discarded round's.
func (h *Harness) runProviderRound(
	ctx context.Context,
	preq ProviderRequest,
	sink EventSink,
	ssink *stampSink,
	now func() time.Time,
	round *RoundTiming,
) (ProviderResult, error) {
	ssink.roundReset()
	callStart := now()
	presult, err := h.provider.RunTurn(ctx, preq, sink)
	callEnd := now()
	if first := ssink.takeFirstEvent(); !first.IsZero() {
		round.StartupToFirstEventSecs = first.Sub(callStart).Seconds()
		round.StreamSecs = callEnd.Sub(first).Seconds()
	} else {
		// The provider emitted no events this round (e.g. a tool-call-only
		// round from a direct-API provider). There was never a first event
		// to split on, so attribute the full call duration to startup and
		// leave StreamSecs 0 — the whole wait was pre-stream latency.
		round.StartupToFirstEventSecs = callEnd.Sub(callStart).Seconds()
		round.StreamSecs = 0
	}
	return presult, err
}

// enforceToolCallContract is the post-provider tool-call contract step
// (NEX-581). After the provider returns and before tool execution, it
// guarantees the consumer never sees a malformed/leaked tool call:
//
//  1. Detect: scan the surfaced text + tool-call args for leaked protocol
//     tokens (the harmony/channel/<|tool_call|> class the gemma A/B caught)
//     or a tool-call emitted as literal text.
//  2. No leak → return presult untouched. The clean-provider path is a
//     zero-cost passthrough; existing turns are unaffected.
//  3. Repair: on a leak, attempt a structural repair (strip tokens; extract
//     a tool-call-as-text into a proper ToolInvocation) WITHOUT a model
//     round trip. A clean repair is applied to presult and returned.
//  4. Retry: under repair-then-retry strictness, if repair couldn't recover
//     a clean result, retry the round ONCE with a tightened system nudge,
//     reusing runProviderRound so the retried round OVERWRITES this round's
//     timing entry and its usage REPLACES (is not added to) the bad round's.
//     The retried result itself runs back through detect→repair so a
//     second-time leak is still cleaned before surfacing.
//  5. Surface flagged: when retry is exhausted (or strictness is tolerant),
//     surface the best-effort repaired result, flagged via a ToolCallRepaired
//     event.
//
// Emits ToolCallRepaired{detected|repaired|retried|flagged} for observability.
func (h *Harness) enforceToolCallContract(
	ctx context.Context,
	preq ProviderRequest,
	presult ProviderResult,
	sink EventSink,
	ssink *stampSink,
	now func() time.Time,
	round *RoundTiming,
) (ProviderResult, error) {
	hasTools := len(preq.Tools) > 0
	// CRITICAL (NEX-745 review gate): the tool-call-as-text
	// EXTRACTION/repair path must be gated off whenever the provider owns
	// its own tool execution (preq.ToolExecutor != nil — wired by run.go
	// for subprocess-stream+SupportsCustomTools providers like claudesdk,
	// see the wiring comment above). Those providers' round-trip
	// completes INSIDE RunTurn, so ProviderResult.ToolCalls is empty BY
	// DESIGN, not because a call leaked as text — detectLeak/repairLeak's
	// hasTools-gated "JSON tool-call in prose" branch has no way to tell
	// the two apart on its own. Without this gate, a model recapping a
	// tool it already (really) called — or an attacker-injected
	// {"name":...,"arguments":...} blob in untrusted content — gets
	// fabricated into a ToolInvocation and EXECUTED for real, with no
	// sidecar round trip and no gate: a prompt-injection ->
	// arbitrary-tool-execution path. Passing hasTools=false to
	// detectLeak/repairLeak here disables ONLY that extraction branch
	// (protocol-token stripping still runs); it does not change behavior
	// for any other provider, since ToolExecutor is nil everywhere else.
	extractGate := hasTools && preq.ToolExecutor == nil
	report := detectLeak(presult, extractGate)
	if !report.detected {
		return presult, nil // clean — zero-cost passthrough
	}
	sink.Emit(ToolCallRepaired{Stage: toolCallStageDetected, Detail: report.detail})

	repaired := repairLeak(presult, extractGate)
	if repaired.clean {
		sink.Emit(ToolCallRepaired{Stage: toolCallStageRepaired, Detail: repaired.detail})
		return applyRepair(presult, repaired), nil
	}

	// Repair couldn't fully recover. Under tolerant strictness, accept the
	// best-effort repaired result without a retry round — surface flagged.
	if effectiveStrictness(preq.ToolCallStrictness) == ToolCallStrictnessTolerant {
		sink.Emit(ToolCallRepaired{Stage: toolCallStageFlagged, Detail: repaired.detail})
		return applyRepair(presult, repaired), nil
	}

	// Strict: retry the round ONCE with a tightened instruction. The retry
	// REPLACES the bad round — overwrite this round's timing and use only the
	// retried usage.
	retryReq := preq
	retryReq.AppendSystemPrompt = preq.AppendSystemPrompt + retryNudge
	sink.Emit(ToolCallRepaired{Stage: toolCallStageRetried, Detail: report.detail})

	retryResult, err := h.runProviderRound(ctx, retryReq, sink, ssink, now, round)
	if err != nil {
		// Return the pre-retry presult (the leak-detected round's
		// already-billed usage/text) rather than a zeroed
		// ProviderResult — RunStep's caller uses this partial result
		// (see run_step.go); RunTurn's own error branch only reads err
		// here (its finalText/allInvocations/totalUsage come from prior
		// loop rounds, never from this return value), so this change is
		// a no-op for RunTurn's behavior.
		return presult, err
	}

	// Re-run detect→repair on the retried result. Cap at one retry: whatever
	// the second round produced is what we surface (clean, or flagged).
	// retryReq carries the same ToolExecutor as preq, so the extraction
	// gate above applies identically here.
	retryReport := detectLeak(retryResult, extractGate)
	if !retryReport.detected {
		return retryResult, nil // retry produced a clean turn
	}
	sink.Emit(ToolCallRepaired{Stage: toolCallStageDetected, Detail: retryReport.detail})
	retryRepaired := repairLeak(retryResult, extractGate)
	stage := toolCallStageRepaired
	if !retryRepaired.clean {
		stage = toolCallStageFlagged
	}
	sink.Emit(ToolCallRepaired{Stage: stage, Detail: retryRepaired.detail})
	return applyRepair(retryResult, retryRepaired), nil
}

// applyRepair folds a repairOutcome back into a ProviderResult, preserving
// the provider's usage/session/model fields (only the surfaced text and
// tool calls change). Usage and timing are NOT touched here — a repaired
// (non-retried) result keeps the original round's usage, since no extra
// model call was made.
func applyRepair(res ProviderResult, r repairOutcome) ProviderResult {
	res.FinalText = r.text
	res.ToolCalls = r.toolCalls
	return res
}

// executeToolCall runs one tool invocation through the BeforeToolCall →
// runner/MCP dispatch → AfterToolCall pipeline. Returns the resulting
// tool_result ProviderMessage to append to the next request, the
// completed ToolInvocation to record in allInvocations, an aborted
// flag (true when a hook returned HookAbort or ctx was cancelled
// mid-call), and any hook error. Runner-level errors are captured
// onto the ToolCallResult and don't return err — the model should see
// the error string and decide what to do.
func (h *Harness) executeToolCall(
	ctx context.Context,
	inv ToolInvocation,
	stepCount int,
	runner ToolRunner,
	mcpClient *mcpclient.Client,
	sink EventSink,
) (toolMsg ProviderMessage, completed ToolInvocation, aborted bool, err error) {
	call := ToolCall{ID: inv.ID, Name: inv.Name, Args: inv.Args}

	// BeforeToolCall hook.
	btc := BeforeToolCallCtx{Call: call, Step: stepCount + 1}
	btc, aborted, err = h.hooks.runBeforeToolCall(ctx, btc)
	if err != nil || aborted {
		return
	}
	call = btc.Call

	// Per-call deny: a BeforeToolCall hook set Deny=true (returning
	// HookContinue). Skip runner.Run/MCP entirely, hand the model the
	// hook-supplied Result/Err as the tool_result, still fire
	// AfterToolCall (audit), and continue the loop — do NOT abort.
	// Mirrors the normal-path tcr→toolMsg→completed construction below.
	if btc.Deny {
		sink.Emit(ToolCallStart{ID: call.ID, Name: call.Name, Args: call.Args})

		resultJSON := btc.Result
		if resultJSON == nil {
			resultJSON = json.RawMessage(`null`)
		}
		tcr := ToolCallResult{ID: call.ID, Result: resultJSON, Err: btc.Err}
		sink.Emit(tcr)

		// AfterToolCall hook — fires on denied calls so observability
		// sees the refusal.
		atc := AfterToolCallCtx{Call: call, Result: tcr, Step: stepCount + 1}
		atc, aborted, err = h.hooks.runAfterToolCall(ctx, atc)
		if err != nil || aborted {
			return
		}

		completed = ToolInvocation{
			ID:     call.ID,
			Name:   call.Name,
			Args:   call.Args,
			Result: atc.Result.Result,
			Err:    atc.Result.Err,
		}

		resultStr := string(atc.Result.Result)
		if atc.Result.Err != "" {
			resultStr = fmt.Sprintf("error: %s", atc.Result.Err)
		}
		toolMsg = ProviderMessage{
			Role:       "tool_result",
			Content:    resultStr,
			ToolCallID: call.ID,
			ToolName:   call.Name, // required by Gemini's FunctionResponse contract; ignored by other providers
		}
		return
	}

	sink.Emit(ToolCallStart{ID: call.ID, Name: call.Name, Args: call.Args})

	if ctx.Err() != nil {
		aborted = true
		return
	}

	var resultJSON json.RawMessage
	var runErr error
	if mcpClient != nil && mcpClient.IsMCPTool(call.Name) {
		resultJSON, runErr = mcpClient.Call(ctx, call.Name, call.Args)
	} else {
		resultJSON, runErr = runner.Run(ctx, call)
	}
	var toolErrStr string
	if runErr != nil {
		toolErrStr = runErr.Error()
		resultJSON = json.RawMessage(`null`)
	}

	tcr := ToolCallResult{ID: call.ID, Result: resultJSON, Err: toolErrStr}
	sink.Emit(tcr)

	// AfterToolCall hook.
	atc := AfterToolCallCtx{Call: call, Result: tcr, Step: stepCount + 1}
	atc, aborted, err = h.hooks.runAfterToolCall(ctx, atc)
	if err != nil || aborted {
		return
	}

	completed = ToolInvocation{
		ID:     call.ID,
		Name:   call.Name,
		Args:   call.Args,
		Result: atc.Result.Result,
		Err:    atc.Result.Err,
	}

	resultStr := string(atc.Result.Result)
	if atc.Result.Err != "" {
		resultStr = fmt.Sprintf("error: %s", atc.Result.Err)
	}
	toolMsg = ProviderMessage{
		Role:       "tool_result",
		Content:    resultStr,
		ToolCallID: call.ID,
		ToolName:   call.Name, // required by Gemini's FunctionResponse contract; ignored by other providers
	}
	return
}

// toolExecutorAdapter implements ToolExecutor (NEX-745 §4) by delegating
// to the harness's own executeToolCall — the exact BeforeToolCall →
// runner/MCP dispatch → AfterToolCall pipeline direct-api providers use.
// Constructed once per turn in runTurn and handed to a subprocess-stream
// provider (claudesdk) via ProviderRequest.ToolExecutor; the provider
// calls Execute synchronously whenever its subprocess reports a custom
// tool call mid-stream, then feeds the result back over its own wire.
//
// step is a local monotonic counter (NOT the outer runTurn loop's
// stepCount): a claudesdk turn's entire tool round-trip happens inside
// ONE call to provider.RunTurn, so there is no "round N" to key off of
// here. It only has to be unique-and-increasing within the turn for the
// BeforeToolCallCtx/AfterToolCallCtx.Step field hooks observe.
type toolExecutorAdapter struct {
	h         *Harness
	runner    ToolRunner
	mcpClient *mcpclient.Client
	sink      EventSink
	step      atomic.Int64
}

func (a *toolExecutorAdapter) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	step := int(a.step.Add(1))
	inv := ToolInvocation{ID: call.ID, Name: call.Name, Args: call.Args}
	_, completed, aborted, err := a.h.executeToolCall(ctx, inv, step-1, a.runner, a.mcpClient, a.sink)
	if err != nil {
		return ToolResult{}, err
	}
	if aborted {
		if ctx.Err() != nil {
			return ToolResult{}, ctx.Err()
		}
		return ToolResult{}, context.Canceled
	}
	return ToolResult{Result: completed.Result, Err: completed.Err}, nil
}

func lowerRequest(req TurnRequest) ProviderRequest {
	var messages []ProviderMessage

	for _, e := range req.SessionTail {
		// NEX-320 cross-turn: carry thinking blocks attached to assistant
		// SessionEvents into the rebuilt ProviderMessage so claude's
		// toClaudeMessages can re-emit them. Anthropic API rejects
		// multi-turn requests whose assistant history is missing the
		// thinking blocks from prior thinking-mode turns.
		//
		// NEX-340 cross-turn: same shape on the openai side for DeepSeek
		// reasoner-style models — carry reasoning_content so the openai
		// provider's toOpenAIMessages can re-emit it.
		//
		// Tool-use cross-turn: a prior turn's assistant tool_use blocks
		// were flattened into one SessionEvent per call (RawJSON-bearing).
		// Strict providers (OpenAI/DeepSeek, Bedrock) require those
		// reconstituted as a single assistant message with structured
		// tool_calls, not N separate messages, and require tool_results
		// keyed by the original call id. Coalesce consecutive assistant
		// events into one ProviderMessage; parse RawJSON into ToolCalls
		// per the event's Provider shape.
		switch e.Role {
		case RoleAssistant:
			tc, hasToolCall := parseSessionToolCall(e)
			if n := len(messages); n > 0 && messages[n-1].Role == "assistant" {
				last := &messages[n-1]
				if e.Content != "" {
					if last.Content != "" {
						last.Content += "\n"
					}
					last.Content += e.Content
				}
				if len(e.ThinkingBlocks) > 0 {
					last.ThinkingBlocks = append(last.ThinkingBlocks, e.ThinkingBlocks...)
				}
				if e.ReasoningContent != "" && last.ReasoningContent == "" {
					last.ReasoningContent = e.ReasoningContent
				}
				if hasToolCall {
					last.ToolCalls = append(last.ToolCalls, tc)
				}
				continue
			}
			msg := ProviderMessage{
				Role:             "assistant",
				Content:          e.Content,
				ThinkingBlocks:   e.ThinkingBlocks,
				ReasoningContent: e.ReasoningContent,
			}
			if hasToolCall {
				msg.ToolCalls = []ToolInvocation{tc}
			}
			messages = append(messages, msg)
		case RoleTool:
			messages = append(messages, ProviderMessage{
				Role:       "tool_result",
				Content:    e.Content,
				ToolCallID: e.ToolCallID,
			})
		default:
			messages = append(messages, ProviderMessage{
				Role:    string(e.Role),
				Content: e.Content,
			})
		}
	}

	if len(req.Inbox) > 0 {
		// Format inbox items with their msg_id so the funnel can wire
		// id-referencing contracts (e.g. triage(msg_id=...)) on top.
		// bridle stays out of those policies — it only renders the
		// listing. Items with MsgID==0 are synthetic/internal.
		content := "Messages received since last turn:\n"
		for _, item := range req.Inbox {
			if item.MsgID > 0 {
				content += fmt.Sprintf("[msg_id=%d from=%s]: %s\n", item.MsgID, item.From, item.Content)
			} else {
				content += fmt.Sprintf("[from %s]: %s\n", item.From, item.Content)
			}
		}
		messages = append(messages, ProviderMessage{Role: "user", Content: content})
	}

	if req.UserMessage != "" {
		messages = append(messages, ProviderMessage{Role: "user", Content: req.UserMessage})
	}

	return ProviderRequest{
		AspectID:           req.AspectID,
		AppendSystemPrompt: req.AppendSystemPrompt,
		SystemPromptMode:   req.SystemPromptMode,
		Session:            req.Session,
		Messages:           messages,
		Tools:              req.Tools,
		ToolChoice:         req.ToolChoice,
		MCP:                req.MCP,
		MaxSteps:           req.MaxSteps,
		Model:              req.Model,
		Cwd:                req.Cwd,
		ProviderEnv:        req.ProviderEnv,
		// NEX-299 Pass 2: sampling / output controls flow through to
		// the provider, which translates each to its wire format and
		// silently ignores fields its API doesn't support.
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		TopK:            req.TopK,
		Seed:            req.Seed,
		MaxOutputTokens: req.MaxOutputTokens,
		StopSequences:   req.StopSequences,
		ResponseFormat:  req.ResponseFormat,
		// Claude-only extended-thinking budget; other providers ignore it.
		ThinkingBudgetTokens: req.ThinkingBudgetTokens,
		// NEX-581: per-aspect tool-call contract strictness flows to the
		// harness's post-provider repair/retry step.
		ToolCallStrictness: req.ToolCallStrictness,
		// Context contract (NEX-581): the per-aspect context-window
		// policy flows to the provider, which maps TargetWindow to its
		// engine knob (ollama num_ctx) or no-ops it (fixed-window APIs).
		// PromptBudget is enforced at the harness seam below, not here.
		ContextPolicy: req.ContextPolicy,
	}
}

// parseSessionToolCall reconstructs a ToolInvocation from an assistant
// SessionEvent's RawJSON when the event represents a prior-turn tool_use
// block. Returns (zero, false) when the event isn't a parseable tool
// call OR when bridle doesn't yet know the provider's RawJSON shape.
//
// Used by lowerRequest to rebuild structured ToolCalls on the cross-
// turn replay path — without it, the rebuilt assistant ProviderMessage
// has neither Content nor ToolCalls, and providers reject it as
// malformed.
//
// Per-provider RawJSON shape is set by the provider's extractResult /
// equivalent; the unmarshal targets here must stay in sync with that.
// Today only ProviderOpenAI is wired — the rest fall through to false
// (the cross-turn tool replay path isn't exercised for them yet; add
// when needed).
func parseSessionToolCall(e SessionEvent) (ToolInvocation, bool) {
	if e.Role != RoleAssistant || len(e.RawJSON) == 0 {
		return ToolInvocation{}, false
	}
	switch e.Provider {
	case ProviderOpenAI:
		var tc struct {
			ID       string `json:"id"`
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		}
		if err := json.Unmarshal(e.RawJSON, &tc); err != nil || tc.Function.Name == "" {
			return ToolInvocation{}, false
		}
		return ToolInvocation{
			ID:   tc.ID,
			Name: tc.Function.Name,
			Args: json.RawMessage(tc.Function.Arguments),
		}, true
	case ProviderClaude:
		// Claude stores the marshaled anthropic.ToolUseBlock:
		// { "id":..., "name":..., "input": {...}, "type": "tool_use" }
		var tu struct {
			Type  string          `json:"type"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(e.RawJSON, &tu); err != nil || tu.Type != "tool_use" || tu.Name == "" {
			return ToolInvocation{}, false
		}
		return ToolInvocation{
			ID:   tu.ID,
			Name: tu.Name,
			Args: tu.Input,
		}, true
	}
	return ToolInvocation{}, false
}

// promptBytes is the marshaled size of the round's request messages —
// the request-size lens for TurnTiming. Returns -1 on a marshal error
// (e.g. invalid RawMessage payloads); instrumentation never fails the
// turn.
func promptBytes(msgs []ProviderMessage) int {
	b, err := json.Marshal(msgs)
	if err != nil {
		return -1
	}
	return len(b)
}

// promptText concatenates the round's input text — the system prompt
// plus every message's content — into a single string for the usage
// estimator (NEX-581). It is an approximation of the input the model
// received: tool-schema and structural framing aren't counted, but the
// prose dominates the token count and this is only a last-resort floor
// for engines that report no usage at all.
func promptText(req ProviderRequest) string {
	var b strings.Builder
	if req.AppendSystemPrompt != "" {
		b.WriteString(req.AppendSystemPrompt)
		b.WriteByte('\n')
	}
	for _, m := range req.Messages {
		if m.Content != "" {
			b.WriteString(m.Content)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func addUsage(a, b Usage) Usage {
	return Usage{
		InputTokens:              a.InputTokens + b.InputTokens,
		OutputTokens:             a.OutputTokens + b.OutputTokens,
		CacheReadInputTokens:     a.CacheReadInputTokens + b.CacheReadInputTokens,
		CacheCreationInputTokens: a.CacheCreationInputTokens + b.CacheCreationInputTokens,
		CostUSD:                  a.CostUSD + b.CostUSD,
		ReasoningTokens:          a.ReasoningTokens + b.ReasoningTokens,
		// If ANY round contributed an estimated count, the turn total is
		// approximate — flag it so cost accounting knows not to trust it
		// as exact.
		Estimated: a.Estimated || b.Estimated,
	}
}

func partialAbort() TurnResult {
	return TurnResult{StopReason: StopReasonAborted}
}

func partialAbortWith(text string, invocations []ToolInvocation, steps int, usage Usage) TurnResult {
	return TurnResult{
		FinalText:  text,
		ToolCalls:  invocations,
		StepCount:  steps,
		Usage:      usage,
		StopReason: StopReasonAborted,
	}
}

func panicErr(r any) error {
	if err, ok := r.(error); ok {
		return err
	}
	return fmt.Errorf("panic: %v", r)
}

// lowerMCPConfig converts a bridle MCPClientConfig to the internal mcpclient ServerSpec slice.
func lowerMCPConfig(cfg *MCPClientConfig) []mcpclient.ServerSpec {
	if cfg == nil {
		return nil
	}
	specs := make([]mcpclient.ServerSpec, 0, len(cfg.Servers))
	for _, s := range cfg.Servers {
		specs = append(specs, mcpclient.ServerSpec{
			Name:      s.Name,
			Transport: mcpclient.Transport(s.Transport),
			Command:   s.Command,
			URL:       s.URL,
			Env:       s.Env,
			Header:    s.Header,
		})
	}
	return specs
}

// mergeToolSurface merges explicit ToolDefs with MCP-loaded ToolDefs, checking
// for name collisions. Returns ErrToolNameCollision on a duplicate name.
func mergeToolSurface(explicit []ToolDef, mcpTools []mcpclient.ToolDef) ([]ToolDef, error) {
	seen := make(map[string]struct{}, len(explicit))
	for _, t := range explicit {
		seen[t.Name] = struct{}{}
	}
	merged := make([]ToolDef, len(explicit), len(explicit)+len(mcpTools))
	copy(merged, explicit)
	for _, t := range mcpTools {
		if _, dup := seen[t.Name]; dup {
			return nil, ErrToolNameCollision
		}
		seen[t.Name] = struct{}{}
		merged = append(merged, ToolDef{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	// MCP servers are re-connected and re-listed on EVERY RunTurn (see the
	// connect/list loop above), and each server's ListTools order is
	// server-dependent (not guaranteed stable across calls, let alone
	// across servers). The merged tool surface is serialized into the
	// prompt prefix on every provider call (provider/openai/openai.go),
	// so any reordering here invalidates the vLLM prefix cache from
	// token 0. Sorting by name makes the serialized block byte-stable
	// across turns regardless of MCP listing order, so the shared
	// prefix (system + tools) can actually be cached.
	sortToolDefsByName(merged)
	return merged, nil
}

// sortToolDefsByName sorts tools in place by Name — the single
// definition of "byte-stable tool-surface ordering" both the
// RunTurn/MCP merge path (mergeToolSurface, above) and the Stream
// direct-api path (lowerStreamToProviderRequest, stream.go) apply, so
// the two paths can't silently diverge on ordering.
func sortToolDefsByName(tools []ToolDef) {
	sort.Slice(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })
}
