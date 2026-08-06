# AA-Omniscience × ctxmap — knowledge/hallucination arms (experiment spec)

**Status:** DRAFT for operator review · **Date:** 2026-07-17 · **Class:** research spike (separate from MVP work per the research/MVP split rule)
**Prior art:** wset curation arms (PR #78 — arm F 12/12 vs 4/12 bare on coding; 4–20× cheaper). This is the knowledge-side analogue, adding the axis the coding arms never measured: **hallucination cost**.

## 1. Hypothesis

The ctxmap harness's ground-or-abstain discipline improves a model's *Omniscience Index* — a metric whose asymmetry (correct +1, hallucination −1, abstention 0) rewards exactly the behavior ctxmap curation enforces. Three separable claims:

- **H1 (harness honesty):** the ctxmap prompt/curation harness alone — empty map, closed book — converts wrong guesses into abstentions (Index rises by not losing points), without materially reducing correct answers.
- **H2 (recall lift):** a seeded map adds correct answers on covered topics (true open-book lift), with hallucination rate not rising on *uncovered* topics.
- **H3 (engine vs scale):** the engine lift (H1+H2) on a small sovereign model (Ornith) recovers a meaningful fraction of the gap to a frontier model (Kimi K3) run bare.

## 2. Benchmark

**AA-Omniscience** (Artificial Analysis, arXiv 2511.13029): 6,000 questions, 42 topics, 6 domains. Score = Omniscience Index ∈ [−100, 100]: +correct, −hallucinated (wrong answer), 0 abstain. At launch nearly all frontier models scored < 0.

**We use the public 10% subset**: `ArtificialAnalysis/AA-Omniscience-Public` on HuggingFace — 600 questions with reference answers. **Gate 0 (before any run): verify the dataset license permits local evaluation use and record it in the results doc.** The 90% holdout stays theirs; our absolute numbers are NOT leaderboard-comparable (different grader, subset) — all conclusions rest on **deltas between our own arms**, graded identically.

## 3. Arms

| Arm | Model | Harness | Book | Measures |
|-----|-------|---------|------|----------|
| K-bare | kimi-k3 (litellm→OpenRouter) | none (direct QA prompt) | closed | baseline; sanity vs published K3 number |
| K-ctx0 | kimi-k3 | ctxmap, **empty map** | closed | H1: honesty pressure alone |
| K-ctxS | kimi-k3 | ctxmap, **seeded** | open (5 topics) | H2: recall lift + off-topic hallucination guard |
| O-bare | ornith (litellm) | none | closed | sovereign baseline |
| O-ctx0 | ornith | ctxmap, empty | closed | H1 at small scale |
| O-ctxS | ornith | ctxmap, seeded | open (5 topics) | H3: engine-vs-scale |

Optional 7th arm (cheap, decide at run time): deepseek-v4-flash bare, as a mid-scale reference point already in the selector table.

**Answer contract (all arms, identical):** the model must end with exactly one of `ANSWER: <text>` or `ABSTAIN`. The ctxmap arms additionally follow the harness's grounding rule (cite recalled context or abstain). Same decoding params across arms within a model; params pinned in the results doc.

## 4. Seeding (K/O-ctxS arms only)

- **Topics (5, pilot):** Software Engineering + 4 chosen after Gate 0 by picking topics whose public-subset questions have ≥25 items and a clean license-safe corpus source (Wikipedia category dumps; CC-BY-SA, attribution recorded).
- **Ingestion:** corpus → ctxmap store via the existing extractor/embed path (`ctxmap/extractor`, `ctxmap/embed`); no bespoke pipeline. Budget cap: ≤ 200MB raw text total (pilot, not a knowledge-base build).
- **Honesty rule:** seeding happens **before** looking at any question text. Nobody curates the corpus against the question set — topic label is the only targeting. (Otherwise we're benchmarking our ability to cheat.)

## 5. Grading

- **Grader:** one local judge for ALL arms (route: the litellm judge lane; model pinned in results doc), replicating the paper's equality-grading rubric: correct / incorrect / abstained. Judge never sees which arm produced the answer.
- **Judge validation gate:** before accepting results, 60 stratified answers hand-labeled by the operator or shadow; judge agreement must be ≥ 90% or the judge model/rubric is revised and ALL arms regraded. (Grid lesson: judge drift invalidates cells silently.)
- **Metrics per arm × topic:** Index, correct %, hallucination %, abstention %, tokens in/out, $ cost, wall-clock. For ctxS arms additionally: covered-topic vs uncovered-topic split (H2's guard is that uncovered-topic hallucination doesn't rise vs ctx0).

## 6. Infrastructure

- Runner: the wset arm harness pattern in bridle (one runner script per arm, resumable, results as JSONL + a summary table checked into the results doc). No new framework.
- Routes: `kimi-k3` (staged 2026-07-17; roll pending pool-idle), `ornith`, judge lane. **Never roll litellm mid-arm** (grid lesson, standing).
- Launch-day caveat: OpenRouter→Moonshot is currently 429-heavy; runner needs bounded retry with backoff and a resume file, and arms record retry counts (a high-retry arm's latency numbers are marked unclean).
- Cost ceiling (approval to spend): K3 arms ≈ 600 q × 3 arms × (~1.5k in / ~400 out avg incl. harness) ≈ $8–15 total at $3/$15 with cache-hit prefix reuse; Ornith arms free. Hard cap $25 — runner aborts if OpenRouter spend exceeds it.

## 7. Acceptance (observable)

- AC1: results JSONL per arm committed, with per-question {id, arm, raw answer, judge verdict}; counts reconcile with the summary table.
- AC2: judge-validation transcript committed showing ≥90% agreement on the 60-item hand-labeled set.
- AC3: the summary table reports Index per arm with 95% bootstrap CIs; H1/H2/H3 each get a one-line verdict tied to the numbers (supported / not supported / inconclusive).
- AC4: seeding manifest committed (corpus sources, sizes, licenses, ingestion timestamps) proving seed-before-questions ordering (timestamps precede first run).
- AC5: total OpenRouter spend for the experiment ≤ $25, shown from the OpenRouter usage dashboard.

## 8. Out of scope

Leaderboard claims; full 6k set; >5 seeded topics; tuning ctxmap parameters per-topic (one global config, the shipping one); fine-tuning anything; using benchmark questions to improve ctxmap (one-way street: the bench measures, never trains).

## 9. Open decisions (operator)

1. Judge model for §5 (recommend: the existing gemma judge lane for zero cost; fallback deepseek-v4-flash-fast if agreement gate fails).
2. The optional deepseek 7th arm — include?
3. Spend approval for the ≤$25 K3 budget (AC5).
