# ctxmap

Cross-referenced working memory for LLM harnesses: a small CPU model distills
each conversation turn into provenance-mandatory facts with a trust lifecycle;
the harness assembles a compact prompt (verified core + relevant subgraph +
short tail) instead of replaying the transcript, and serves `recall`/`inspect`
tools backed by the store.

Validated on the agora research harness: **100% vs guess-rate** on probes whose
facts were truncated out of every context, at **~40–90× fewer tokens at the
moment of answering**. Full write-up (idea, architecture, measured results,
findings): [agora `docs/ctxmap.md`](https://github.com/CarriedWorldUniverse/agora/blob/ctxmap-harness/docs/ctxmap.md) — agora is the research harness where this was validated and remains the evaluation bench.

## Packages

- `store` — SQLite fact store: provenance-mandatory construction, trust-ranked
  contradiction resolution, reuse-confirmation promotion, session-scoped
  visibility (VERIFIED = project-wide, PROPOSED = session-local), secret
  deny-pass, invariant audit.
- `render` — epoch-frozen core block + per-turn subgraph; recall rendering.
- `memory` — the engine a host harness drives: `AssembleBlocks` →
  (host model call) → `RecordTurn`; async extraction + reconciliation;
  `Tools()` for recall/inspect. Pure Go.
- `extractor`, `embed` — seam interfaces (pure Go) + llama.cpp-backed
  implementations behind the `ctxmap_llama` build tag (Qwen3-1.7B extract,
  Qwen3-4B kind/source/pair judgment, nomic-embed topic gating; ~11GB RSS,
  CPU-only, in-process).

## Build

    go build ./...                      # pure Go: engine + store + seams
    make vendor-llama                   # stage llama.cpp libs (once)
    go build -tags ctxmap_llama ./...   # with the in-process models

Host integration contract is documented in `memory/engine.go`.
