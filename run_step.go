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
// RunStep has the same recover() boundary Harness.RunTurn has (see
// harness.go's RunTurn): an unrecovered panic here would otherwise
// propagate straight out of Registry.Stream's goroutine and crash the
// whole process — every in-flight turn across every lane, not just the
// offending one (RunStep/Stream is the seam every model call routes
// through, so its blast radius is far broader than RunTurn's own
// panic-isolated funnel path).
func (h *Harness) RunStep(ctx context.Context, preq ProviderRequest, sink EventSink) (presult ProviderResult, err error) {
	now := h.clock()
	ssink := &stampSink{inner: sink, now: now}
	sink = ssink

	defer func() {
		if r := recover(); r != nil {
			e := panicErr(r)
			// Stamp directly: this emit bypasses ssink's own stamping
			// (the panic unwound past it), so TS would otherwise be zero.
			sink.Emit(TurnError{Err: e, Stage: TurnErrorStageHarnessRecover, TS: now()})
			presult = ProviderResult{}
			err = e
		}
	}()

	var round RoundTiming
	presult, err = h.runProviderRound(ctx, preq, sink, ssink, now, &round)
	if err != nil {
		return presult, err
	}

	// NEX-581 tool-call contract (run.go's enforceToolCallContract doc):
	// on a retry-round error, this returns the pre-retry presult (the
	// leak-detected round's already-billed usage/text) rather than a
	// zeroed ProviderResult — matching RunTurn's "return the partial
	// result" convention on its own analogous error paths (run.go ~146).
	presult, err = h.enforceToolCallContract(ctx, preq, presult, sink, ssink, now, &round)
	if err != nil {
		return presult, err
	}

	// Usage contract (NEX-581, usage.go): RunTurn normalizes every
	// round's usage before folding it into the turn total (run.go ~178)
	// so a completed turn never reports silently-zero usage. RunStep is
	// single-round, so this is the turn total — apply the same floor
	// here so the Stream path (which reuses RunStep for direct-api
	// lanes) holds the same invariant RunTurn does.
	presult.Usage = normalizeUsage(presult.Usage, promptText(preq), presult.FinalText)

	return presult, nil
}
