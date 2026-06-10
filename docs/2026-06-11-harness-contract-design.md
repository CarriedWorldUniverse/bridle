# bridle owns the contract: engine-agnostic tool-call, usage, and context

**Status:** spec (operator-directed; NEX-581 core — "expand bridle into our own complete harness")
**Date:** 2026-06-11
**Driver:** shadow (with the operator)
**Grounded in:** `docs/2026-06-11-gemma-serving-evaluation.md` + the live ollama-vs-vLLM A/B (2026-06-11) + the turn-timing baseline.

## The idea

The operator's goal: *"expand bridle so we own a complete harness for the APIs we depend on — control and flavour ours, engines swappable underneath."* The gemma A/B proved why this is the right altitude rather than an engine bet: the *same model* behaved differently per serving path (openai-compat shim leaked Gemma's raw tool tokens + reported zero usage; ollama-native `/api/chat` was 10/10 clean with full usage; vLLM was clean-but-zero-usage and couldn't share the GPU). The lesson: **the engine is an implementation detail; the harness should own the guarantees.** bridle already abstracts providers — this spec makes that abstraction *load-bearing* by giving bridle three engine-agnostic contracts every provider must honor, with bridle filling the gaps an engine leaves.

## The three contracts

### 1. Tool-call contract — clean tool calls, or repair, or retry

**Guarantee:** a turn never surfaces a malformed/leaked tool call to the consumer. Whatever the engine emits, bridle delivers either a well-formed tool call or a clean text turn.

- **Normalize:** every provider's tool-call output is parsed into bridle's canonical `ToolInvocation` shape (already exists). Today this is per-provider; formalize it as a contract every provider's adapter must satisfy.
- **Detect leakage:** bridle scans model output for raw tool/reasoning protocol tokens that escaped the engine's parser (the `<|channel|>thought…`, `<|tool_call|>` class the A/B caught on the compat shim). A leak = a contract violation.
- **Repair-or-retry:** on a detected leak or an unparseable tool call, bridle (a) attempts a structural repair (strip the protocol tokens, re-parse the intended call), and if that fails (b) retries the turn once with a tightened instruction (or a stricter decoding hint where the engine supports it). Only after repair+retry fail does it surface the raw text, flagged. This is the *insurance* that makes a weaker engine usable — the "feels dumb" failures get caught at the harness, not shipped to the operator.
- **Configurable strictness:** per-aspect (a research aspect may tolerate a degraded turn; a builder must not ship a garbled tool call).

### 2. Usage contract — every engine reports tokens, normalized

**Guarantee:** every completed turn has input/output (and cache, where available) token counts in `Usage`, regardless of engine.

- The A/B exposed the gap: ollama-native reports usage 10/10; vLLM-via-bridle's-openai-provider reported 0/10 (needs `stream_options.include_usage`); the openai-compat main-turn path reports zero (NEX-589). These are per-provider holes.
- **Contract:** each provider adapter either (a) extracts real usage from the engine response, (b) requests it explicitly (the `stream_options.include_usage` / equivalent flag), or (c) as a last resort, bridle estimates from a tokenizer and marks the count `estimated=true`. Never silently zero.
- Folds in **NEX-589** (openai-compat usage gap) as one instance of the general contract.
- Feeds the existing `TurnTiming`/`Usage` → TurnFrame path (NEX-561/563) so cost is always visible.

### 3. Context contract — window policy owned by the harness

**Guarantee:** context-window sizing is a bridle-owned, per-aspect policy, expressed once and mapped to whatever knob the engine exposes.

- ollama has `num_ctx` (now wired, NEX-562); vLLM has `--max-model-len`; APIs have model-fixed windows. Today these are scattered.
- **Contract:** a per-aspect context policy (target window, prompt-budget) on the bridle request; each provider maps it to its engine's mechanism (or no-ops where fixed). bridle warns when an assembled prompt approaches the policy budget (ties to the prompt-bytes accounting already in TurnTiming).

## Why this delivers "own the harness"

With these three contracts, the engine choice (ollama / vLLM / direct API / a future local runtime) becomes a swappable backend behind a constant guarantee surface. Switching engines is a config change; the tool-call cleanliness, the usage accounting, and the context policy don't move. The compounding bet from `[[gemma-control-direction]]` is realized here: as bridle's contract layer matures, *every* gemma-lane aspect levels up, and we can adopt a better engine (vLLM for a concurrent roundtable lane, say) without re-proving quality — the harness already guarantees it.

## Architecture (where each lives in bridle)

- Tool-call normalize/leak-detect/repair: a post-provider step in `run.go`'s round loop (after the provider returns, before tool execution), plus a per-provider parse contract. Repair is engine-agnostic (token-stripping + re-parse); retry reuses the existing round machinery.
- Usage normalize: a thin wrapper each provider's result passes through; the estimator is a shared tokenizer helper.
- Context policy: a field on the request + a per-provider `applyContextPolicy` mapping; warning rides the existing prompt-bytes timing.
- All three are provider-interface-level — no consumer (funnel/agora) change required; they just get cleaner turns + reliable usage.

## Non-goals (v1)

- A from-scratch direct-inference harness (the eval's explicit "wrong altitude" — engines stay external).
- Re-litigating the engine choice (settled by the A/B: ollama-native is the steady-state lane; vLLM is shelved-but-configured for a future dedicated concurrent lane).
- Streaming-prose tool-call repair mid-stream (v1 repairs at round boundaries; mid-stream is a later refinement).

## Testing

- Tool-call contract: feed crafted provider outputs (clean / leaked-tokens / unparseable) through the normalize→repair→retry path; assert clean call out, or a flagged text turn after repair+retry exhaust. Leak-corpus from the real `<|channel|>` samples the A/B captured.
- Usage contract: each provider path asserts non-zero usage (real or estimated-flagged); the vLLM/openai `include_usage` and the compat-shim gap (NEX-589) get explicit tests.
- Context contract: policy maps to num_ctx (ollama) and is a no-op for fixed-window providers; budget-approach warning fires.
- Engine-swap: the same turn through two providers yields the same contract guarantees (clean call, usage present).

## Sequencing

1. **Usage contract + NEX-589** — smallest, immediate cost-visibility win; do first.
2. **Tool-call contract** (normalize → leak-detect → repair → retry) — the durable quality guarantee; the meat.
3. **Context contract** — the lightest; folds in.

Each is an independent bridle PR. The keel native-flip (separate, in flight) is the immediate operational fix; this spec is the durable layer beneath it.

## Open questions (operator)

- Repair vs retry default order per aspect class (builders strict-retry; research tolerant-repair?) — proposed defaults above; confirm.
- Estimated-usage acceptable as the last-resort floor, or should a provider that can't report real usage be considered non-conformant (hard-fail)? (Proposed: estimate + flag, never hard-fail.)
