package bridle

import "time"

// context.go implements the context contract (NEX-581): context-window
// sizing is a bridle-owned, per-aspect policy expressed ONCE on the
// request and mapped to whatever knob each engine exposes — so callers
// don't reach past bridle to set engine-specific context flags.
//
// The policy has two independent levers:
//
//   - TargetWindow is the desired context window in tokens. Each
//     provider maps it to its engine's mechanism: ollama sets
//     options.num_ctx; fixed-window API providers (claude / openai /
//     bedrock / gemini) no-op it because the window is model-fixed and
//     not a per-request knob; vLLM-via-openai no-ops it too (the window
//     is server-side, --max-model-len, not per-request). CLI providers
//     no-op it.
//
//   - PromptBudget is a soft cap, in tokens, on the assembled prompt
//     size. It is engine-AGNOSTIC: ALL providers honor it. After
//     request assembly, bridle estimates the assembled prompt's token
//     count (the usage contract's estimateTokens over the assembled
//     input) and, when that meets or exceeds PromptBudget, emits a
//     ContextBudgetWarning for the funnel to log/act on. v1 is
//     warn-only: bridle never truncates or hard-fails (truncation
//     policy is future work).
//
// The zero value is "no policy": no num_ctx override (engine defaults
// hold) and no budget warning. This preserves current behaviour for
// callers that don't set a policy.
type ContextPolicy struct {
	// TargetWindow is the desired context window in tokens. 0 = no
	// preference (engine default / model-fixed window). Providers with a
	// per-request window knob (ollama num_ctx) map this; fixed-window
	// providers no-op it.
	TargetWindow int

	// PromptBudget is a soft cap, in tokens, on the assembled prompt
	// before bridle warns. 0 = no budget (no warning ever). Honored by
	// ALL providers — the warning is computed at the harness seam, not
	// in the provider.
	PromptBudget int
}

// ContextBudgetWarning fires when the assembled prompt's estimated
// token count meets or exceeds the request's ContextPolicy.PromptBudget
// (NEX-581). It is observability only — bridle does NOT truncate or
// hard-fail on it; the funnel decides what to do (log, trim future
// turns, alert). Assembled is the estimated token count of the prompt
// the provider received this round; Budget is the policy cap that was
// crossed. Mirrors MCPServerFailed's stamped-event shape.
type ContextBudgetWarning struct {
	Assembled int       // estimated assembled-prompt token count
	Budget    int       // the ContextPolicy.PromptBudget that was met/exceeded
	TS        time.Time // stamped by the harness at emission; zero outside a harness turn
}

func (ContextBudgetWarning) event() {}
