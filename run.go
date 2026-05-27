package bridle

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/CarriedWorldUniverse/bridle/internal/mcpclient"
)

// runTurn is the inner implementation, called by RunTurn after the panic trap.
func (h *Harness) runTurn(ctx context.Context, req TurnRequest, runner ToolRunner, sink EventSink) (TurnResult, error) {
	// Connect MCP servers and merge tool surface (direct-api providers only).
	var mcpClient *mcpclient.Client
	caps := h.provider.Capabilities()
	if caps.SupportsMCP && req.MCP != nil {
		specs := lowerMCPConfig(req.MCP)
		var err error
		mcpClient, err = mcpclient.Connect(ctx, specs)
		if err != nil {
			return TurnResult{StopReason: StopReasonError}, err
		}
		defer mcpClient.Close()

		mcpTools := mcpClient.Tools()
		merged, err := mergeToolSurface(req.Tools, mcpTools)
		if err != nil {
			return TurnResult{StopReason: StopReasonError}, err
		}
		req.Tools = merged
	}

	// Lower TurnRequest → ProviderRequest.
	preq := lowerRequest(req)

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

		// Run the provider turn.
		presult, err := h.provider.RunTurn(ctx, preq, sink)
		if err != nil {
			sink.Emit(TurnError{Err: err, Stage: TurnErrorStageProvider})
			return TurnResult{
				FinalText:  finalText,
				ToolCalls:  allInvocations,
				StepCount:  stepCount,
				Usage:      totalUsage,
				StopReason: StopReasonError,
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
		totalUsage = addUsage(totalUsage, presult.Usage)
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

		// No tool calls → turn is done.
		if len(presult.ToolCalls) == 0 {
			stopReason = presult.StopReason
			break
		}

		// Execute each tool call.
		var toolMessages []ProviderMessage
		for _, inv := range presult.ToolCalls {
			toolMsg, completed, abortedExec, execErr := h.executeToolCall(ctx, inv, stepCount, runner, mcpClient, sink)
			if execErr != nil {
				return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), execErr
			}
			if abortedExec {
				return partialAbortWith(finalText, allInvocations, stepCount, totalUsage), nil
			}
			allInvocations = append(allInvocations, completed)
			toolMessages = append(toolMessages, toolMsg)
			sessionDelta = append(sessionDelta, SessionEvent{Provider: h.provider.Name(), Role: RoleTool, Content: toolMsg.Content})
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
		preq.Messages = append(preq.Messages, ProviderMessage{
			Role:           "assistant",
			Content:        finalText,
			ToolCalls:      presult.ToolCalls,
			ThinkingBlocks: presult.ThinkingBlocks,
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
	}

	result := TurnResult{
		FinalText:     finalText,
		ToolCalls:     allInvocations,
		StepCount:     stepCount,
		Usage:         totalUsage,
		StopReason:    stopReason,
		ResolvedModel: resolvedModel,
		SessionDelta:  sessionDelta,
	}

	// OnTurnDone hook — may mutate SessionDelta.
	otd := OnTurnDoneCtx{Result: &result}
	h.hooks.runOnTurnDone(ctx, otd) //nolint:errcheck
	sink.Emit(TurnDone{Result: result})
	return result, nil
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

func lowerRequest(req TurnRequest) ProviderRequest {
	var messages []ProviderMessage

	for _, e := range req.SessionTail {
		messages = append(messages, ProviderMessage{
			Role:    string(e.Role),
			Content: e.Content,
		})
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
	}
}

func addUsage(a, b Usage) Usage {
	return Usage{
		InputTokens:              a.InputTokens + b.InputTokens,
		OutputTokens:             a.OutputTokens + b.OutputTokens,
		CacheReadInputTokens:     a.CacheReadInputTokens + b.CacheReadInputTokens,
		CacheCreationInputTokens: a.CacheCreationInputTokens + b.CacheCreationInputTokens,
		CostUSD:                  a.CostUSD + b.CostUSD,
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
	return merged, nil
}
