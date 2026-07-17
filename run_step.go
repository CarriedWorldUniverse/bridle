package bridle

import "context"

// RunStep drives ONE provider round for a direct-API lane, WITHOUT
// owning a tool-execution loop — additive to RunTurn, which stays
// untouched (existing nexus/funnel callers are unaffected).
//
// Why this exists (NEX-767 T7 / agora-spec-bridle §2): RunTurn's round
// loop (run.go) executes tool calls itself and re-invokes the provider
// with synthesized tool_results, because that's what the nexus funnel
// wants. agora's Stream contract wants the OPPOSITE for direct-API
// lanes: hand back complete tool_calls and let agora execute them and
// decide what happens next. RunStep is the single-round primitive that
// makes that possible — it reuses runProviderRound (round timing) and
// enforceToolCallContract (NEX-581 leak detection/repair) so a Stream
// round gets the same protection every RunTurn round gets, MINUS the
// loop itself, MINUS tool execution, MINUS the MaxSteps check (agora
// owns those on its own turn engine).
//
// Subprocess-stream lanes (claude-code) do NOT use RunStep — RunTurn is
// already single-shot for them (run.go's `if !caps.SupportsCustomTools`
// break fires before any tool re-invocation), so Stream calls RunTurn
// unmodified for that category. RunStep is direct-api only.
//
// Callers pass an already-lowered ProviderRequest (Stream builds one
// from its own Request shape — see stream.go) rather than a TurnRequest,
// since RunStep skips everything TurnRequest carries for the
// tool-loop/session/MCP machinery it doesn't run.
func (h *Harness) RunStep(ctx context.Context, preq ProviderRequest, sink EventSink) (ProviderResult, error) {
	now := h.clock()
	ssink := &stampSink{inner: sink, now: now}
	sink = ssink

	var round RoundTiming
	presult, err := h.runProviderRound(ctx, preq, sink, ssink, now, &round)
	if err != nil {
		return ProviderResult{}, err
	}

	presult, err = h.enforceToolCallContract(ctx, preq, presult, sink, ssink, now, &round)
	if err != nil {
		return ProviderResult{}, err
	}

	return presult, nil
}
