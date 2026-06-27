# Bridle optimization: instrument, measure, fix, consolidate

**Status:** spec (approved direction: evidence-first optimization)
**Date:** 2026-06-10
**Driver:** shadow (with the operator)

## Problem

Bridle turns feel slow and their cost is opaque. There is no timing
instrumentation anywhere in the repo (the only `time.Since` is an error
diagnostic in claudecode), events carry no timestamps, and nothing records
what a turn actually sends. Known structural suspects:

- The headless-CLI lane (claudecode, codexcli, geminicli, antigravitycli)
  spawns a **fresh subprocess every turn**; per-turn cost of CLI startup +
  session resume has never been measured. (claudepty already exists as a
  persistent-process counterexample.)
- The ollama provider sends an **empty `Options` map**: no `keep_alive`
  (Ollama's default unloads the model after ~5 idle minutes, so keel pays a
  full gemma reload on the 5090 after any quiet spell) and no `num_ctx`
  (silent truncation at the model's default context window).
- The CLI providers carry byte-identical duplicated plumbing (cancel
  watcher ×3, `mergeEnv` ×2, `buildPrompt` ×2, ~90%-identical JSONL scanner
  skeleton ×3, shared error-classification shape).

Priority lanes (operator-chosen): **headless CLI** (cloud-shadow, builders,
maren) and **local gemma/ollama** (keel). The direct-API lane is out of
scope except where shared seams touch it.

## Approach

Evidence-first: instrument → measure on the live lanes → fix the top
measured costs (plus two no-regret fixes) → consolidate duplication last.
No speculative rewrites; every fix gets a before/after number from the
same instrumentation.

## Phase 1 — Instrumentation

The lasting artifact. All events already flow through one seam
(`sink.Emit()` call sites in run.go; the provider call is the single site
at run.go:66), so timing lands without touching any provider.

- **Event timestamps:** additive `TS time.Time` on the event types in
  events.go, stamped at emission. Existing consumers are unaffected
  (struct-field addition only).
- **`TurnTiming` on `TurnResult`:** per round —
  - `AssemblySecs` (lowerRequest + hook time before the provider call),
  - `StartupToFirstEventSecs` (provider call start → first sink event; for
    the CLI lane this ≈ spawn + CLI startup + session resume + TTFT),
  - `StreamSecs` (first event → provider call return),
  - per-tool durations (harness already brackets ToolCallStart/Result),
  - `TotalSecs`.
- **Request-size accounting** per round: prompt bytes (marshaled provider
  request size), message count, tool-definition count. The token-efficiency
  lens; pairs with the existing `Usage` token fields.
- Clock injected (`now func() time.Time` on Harness, defaulting to
  time.Now) so tests drive timing deterministically.
- **Consumer surfacing:** TurnTiming rides TurnResult through the existing
  `OnTurnDone` hook / `TurnDone` event — the nexus funnel forwards it in
  TurnFrame as a small additive follow-up (separate nexus PR), which puts
  per-turn timing in the agora trace pane and dashboard.

## Phase 2 — Measurement pass (live, dMon)

With Phase 1 deployed to the funnel pods: collect ~10 turns per lane —
keel (ollama/gemma), cloud-shadow (claudecode), one builder (codexcli) —
plus maren (antigravitycli) opportunistically. Produce
`docs/turn-timing-baseline.md` in this repo (dated header inside): per lane —
total, startup+TTFT, stream, tool time, prompt bytes, tokens in/out.
This table picks Phase 3's targets and is the baseline every fix is
measured against.

## Phase 3 — Fixes

Two no-regret items land regardless of measurement:

1. **ollama options exposure:** `KeepAlive` and `NumCtx` fields on the
   ollama Provider (plus a generic `Options map[string]any` passthrough),
   threaded into the ChatRequest. Defaults chosen for the always-on keel
   shape: keep-alive default `30m` (configurable), `num_ctx` explicit
   rather than model-default. Config plumbed through nexus per-aspect provider config
   (existing seam; nexus-side wiring is part of the follow-up PR).
2. **Request-size visibility** is Phase 1 itself; any lane found resending
   outsized context (antigravitycli reassembles full history by design)
   gets a quantified ticket rather than a speculative fix.

Measurement-gated items (decided by the baseline table):

- **CLI spawn cost:** if startup+TTFT dominates (expected), the menu is —
  adopt the persistent claudepty architecture for the claude lane, trim
  per-turn startup work (arg/MCP-config assembly, system-prompt spill),
  or accept the cost where it's small relative to model time. Decision
  recorded in the baseline doc with numbers.
- Anything else the table surfaces (e.g. tool round-trip overhead).

## Phase 4 — Scoped consolidation (last, mechanical)

Extract to `internal/subprocess` (new package):

- cancel watcher (SIGTERM → 5s grace → SIGKILL; byte-identical ×3),
- `mergeEnv` (byte-identical ×2),
- `buildPrompt` (identical ×2; antigravitycli keeps its full-context
  variant),
- JSONL scanner skeleton (`scanJSONLines(r, perLine func(...))`; each
  provider keeps its own event-schema dispatch),
- error-classification table structure (patterns stay per-provider).

Strictly behavior-preserving; the providers' existing test suites passing
unchanged is the acceptance gate. Lands after measurement so refactor noise
can't contaminate the baseline.

## Non-goals

- Direct-API lane optimization (anthropic/openai/bedrock/gemini HTTP).
- Funnel-side context/compaction policy (nexus's job, not bridle's).
- Rewriting any provider's stream schema handling.
- A benchmarking harness/CI perf suite (the baseline doc + live timing in
  TurnFrame is the right weight today).

## Testing

- Phase 1: unit tests with fake clock + fake provider asserting TurnTiming
  fields and event timestamps; existing suites unaffected.
- Phase 3 ollama: request-shape tests (keep_alive/num_ctx present in the
  built ChatRequest).
- Phase 4: providers' existing suites pass unchanged; no new behavior.

## Sequencing

1. PR 1 (bridle): Phase 1 instrumentation.
2. PR 2 (bridle): ollama options exposure.
3. PR 3 (nexus, small): bump bridle; forward TurnTiming in TurnFrame;
   per-aspect config plumbing for keep_alive/num_ctx.
4. Measurement pass on dMon → baseline doc committed.
5. PR 4+ (bridle): measurement-gated fixes, each with before/after numbers.
6. PR last (bridle): internal/subprocess consolidation.
