# Gemma serving-engine evaluation — NEX-569

**Date:** 2026-06-11
**Author:** shadow (for operator review — DO NOT COMMIT yet)
**Decision scope:** which serving engine runs gemma for the Carried World platform,
and the bridle work it implies. Foundation for NEX-581 ("expand bridle into our
own complete harness").

---

## TL;DR / Recommendation

**Stay on ollama for the always-on aspect lane *now*, but move bridle off the
openai-compat shim onto ollama's native `/api/chat` (the lane that already exists,
NEX-562/563), and in parallel stand up the *already-deployed* `gemma-vllm` pod
WITH tool-calling flags as a measured A/B candidate for the concurrent-roundtable
lane.** The decision between them turns on one number we do not yet have:
**Gemma-4 tool-call reliability through each engine's parser on our real ~21-tool
turns.** Both engines have *documented, open* Gemma-4 tool-call bugs as of
2026, so this is an empirical call, not a spec-sheet call.

The two top-weighted criteria split the engines:

- **Tool-calling quality (the actual pain point):** vLLM has a *dedicated*
  `gemma4` tool-call parser + chat template (purpose-built for Gemma 4's custom
  `<|tool_call>`/`<tool_call|>` token protocol). ollama's openai-compat path has
  no Gemma-4-specific parser and relies on a generic template — which is almost
  certainly why we see token leakage (`<|channel|>thought<channel|>` appeared in
  a live probe today) and zero usage reporting. **Edge: vLLM — but only if its
  parser is configured (the current deploy has NO tool flags) and only if the
  open leak bugs don't bite us.**
- **GPU coexistence with forge training (the hard constraint):** ollama loads
  ~8.3 GB and is elastic (unload-on-idle is *available* even if we pinned it);
  vLLM pre-allocates a large, **non-reclaimable** KV pool at `gpu-mem-util=0.92`
  (~22 GB of the 24 GB card) and will starve forge unless we cap it hard or use
  vLLM's sleep/wake offload. **Edge: ollama for elasticity — but vLLM is
  survivable if we cap `--gpu-memory-utilization` low and shrink `--max-model-len`
  to the 8192 the official Gemma-4 tool recipe recommends.**

So: **don't rip out ollama, don't bet the platform on vLLM blind.** Invest the
near-term bridle work in *owning the contract* (native `/api/chat` + usage
reporting + per-aspect num_ctx, all behind bridle's provider interface so the
engine stays swappable), and run a real tool-call-reliability A/B before
committing GPU budget to vLLM as the steady-state engine. This is also the
*correct* answer to "own the harness": **the engine is an implementation detail
behind bridle; the harness owns the tool-call contract, the usage accounting, and
the context policy.** That ownership is what NEX-581 should buy — not a religious
engine choice.

---

## Environment as measured (2026-06-11, dMon)

| Fact | Value | Source |
|---|---|---|
| GPU | RTX 5090, 24463 MiB (24 GB), sm_120/Blackwell | `nvidia-smi` |
| Currently resident | 6463 MiB used by **`voxcpm` TTS** (vessel, pid 8542) — **NOT forge** | `nvidia-smi --query-compute-apps` |
| forge training | not resident right now; `/home/forge` perms-locked | observed |
| **gemma model (both engines)** | `google/gemma-4-12B-it` | both deploy specs |
| ollama variant | QAT **q4_0 GGUF** (`hf.co/google/gemma-4-12B-it-qat-q4_0-gguf`), 7.2 GB on disk, **8.3 GB resident** at 131072 ctx | `ollama list` / `ollama ps` |
| ollama config | `OLLAMA_CONTEXT_LENGTH=131072`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_KEEP_ALIVE=-1` | deploy env |
| ollama image | `ollama/ollama:0.30.5` | deploy |
| **vLLM variant** | on-the-fly **FP8** (`--quantization=fp8 --kv-cache-dtype=fp8`), `--max-model-len=32768`, `--gpu-memory-utilization=0.92` | `gemma-4-12b-vllm.yaml` |
| vLLM status | **scaled to 0**; revision 3 *"successfully progressed"* — it ran fine, then parked | `kubectl describe deploy gemma-vllm` |
| **vLLM tool-calling** | **NOT configured** — args have no `--enable-auto-tool-choice`, no `--tool-call-parser gemma4`, no `--chat-template` | `gemma-4-12b-vllm.yaml` lines 27–38 |
| Why vLLM exists / parked | built as the **local cheap/fallback completions lane** (NEX local-fallback-LLM idea), proved FP8 loads on the 5090, never wired for tools → ollama became the real aspect path → vLLM scaled to 0 | install-plan doc, README |

**Two facts that reframe the brief:**

1. The "6.4 G already used" is **voxcpm TTS**, not forge. forge is the *future*
   hard constraint (it'll saturate the card during training bursts), but the
   steady-state co-resident today is a ~6.4 GB TTS server. Budget for forge as a
   *burst* contender, not a constant.
2. **The existing vLLM deploy never did tool-calling.** Any "vLLM is better at
   tools" claim is untested *here* — the parked pod was a completions lane. A
   fair A/B requires re-deploying vLLM WITH the Gemma-4 tool flags first.

---

## Comparison table (engines × 7 criteria)

Scale: ✅ strong · 🟡 workable/uncertain · ❌ weak. Evidence in the notes column.

| Criterion (weight) | ollama (current) | vLLM (parked, needs tool flags) | llama.cpp / llama-server | direct / own-harness |
|---|---|---|---|---|
| **1. Tool-calling quality** (HIGH) | 🟡 generic template, no Gemma-4 parser; token leakage observed live; openai-compat shim works but is uneven. Native `/api/chat` may improve it but ollama has documented Gemma-4 streaming bugs (tool calls routed to reasoning field). | ✅ **dedicated `gemma4` tool-call parser + chat template** for Gemma-4's custom token protocol; guided decoding (Outlines), regex/grammar/JSON-schema constrained output. ⚠️ **open leak bugs** (#39043, #41967) as of Apr 2026. Best *ceiling*, unproven *floor* on our turns. | 🟡 OpenAI-compat tool support exists but Gemma-4-specific parsing is community-maintained; same template-mismatch class of bug. | ✅ **maximum control** — we own the grammar/parse, can constrain decoding exactly to our 21 tools. ❌ **maximum work**; we'd be reimplementing what vLLM's parser already does. |
| **2. Concurrency** (HIGH — roundtable/convene) | ❌ **serializes** — `OLLAMA_NUM_PARALLEL=2` only; multiple woken aspects queue. Documented: "Ollama doesn't support concurrent inference — it queues." | ✅ **built for batched concurrent serving** (continuous batching, PagedAttention). 3 concurrent reqs ≈ 114 tok/s combined vs ollama queueing; peak 793 TPS vs 41 TPS in one 2026 benchmark. **The roundtable is exactly vLLM's use case.** | ❌ similar single-stream limits to ollama (it's the engine under ollama). | 🟡 whatever we build; batching is hard to get right. |
| **3. Latency (short agent turns)** (MED) | 🟡 warm TTFT **~3.0 s med / 5.1 s p90** (our baseline) — almost all prefill/queue. q4_0 is small/fast to load. No FA3 on Blackwell either way. | 🟡 lower decode latency at load (Marlin/CUDA-graphs/torch.compile; ~30% faster tok/s same model) and far lower p99 under concurrency. Single-stream cold-ish TTFT comparable. FP8 prefill on 131k-class ctx can be heavier than q4_0. | 🟡 comparable to ollama single-stream. | ❓ unknown. |
| **4. Context control** (MED) | ✅ per-aspect `num_ctx` now exposed (native lane, NEX-563); server `OLLAMA_CONTEXT_LENGTH=131072`. Elastic per request. | 🟡 `--max-model-len` is **fixed at pod start** (32768 now; official tool recipe says **8192** for tool workloads). Can't right-size per-aspect without per-pod or runtime knobs; KV pool sized to it. | 🟡 ctx fixed at load like vLLM. | ✅ we decide. |
| **5. GPU coexistence w/ forge** (HIGH) | ✅ **8.3 GB resident, elastic** — could unload on idle (we *chose* keep_alive=-1 for the always-on aspect; that's policy, reversible). Leaves ~16 GB for forge today. Cooperative. | ❌→🟡 **pre-allocates non-reclaimable KV** at `gpu-mem-util=0.92` ≈ 22 GB → **starves forge**. Mitigable: cap util low (e.g. 0.35–0.45) + `--max-model-len 8192` to shrink KV, and/or vLLM **sleep/wake CPU-offload** to vacate GPU during training bursts. Adds orchestration. | ✅ elastic like ollama (same llama.cpp core). | 🟡 we'd build the arbitration. |
| **6. "Own the harness" / control** (HIGH) | ✅ **already behind bridle's provider interface**; native ollama provider gives us keep_alive/num_ctx/options as first-class. The *contract* (tools→parse→usage) is ours to own in bridle regardless. | ✅ also just an OpenAI-compat baseURL behind bridle's openai provider; vLLM's richer response objects (full usage, logprobs) give bridle *more* to own. | ✅ OpenAI-compat behind bridle. | ✅✅ ultimate control, but you're building the parser/batcher/grammar yourself — that's "own the engine," not "own the harness." Usually the wrong altitude. |
| **7. Operational cost** (MED) | ✅ **live and working today**; zero setup; manifest declarative (modulo the keep_alive=-1 that needs committing to carriedworld-cloud). | 🟡 **pod already exists & proven to load** (revision 3 progressed); cost = re-add tool flags + chat template + re-tune GPU caps + A/B. Low-moderate. | ❌ new deploy from scratch; no existing manifest. | ❌❌ largest build; ongoing maintenance of a parser that upstream gives us free. |

---

## Why ollama "feels dumb on the ollama API thing" — root cause

The operator's instinct is correct and the baseline data + live probe explain it:

1. **No Gemma-4-aware tool parsing on the openai-compat path.** Today's lane is
   bridle → *openai provider* → ollama `/v1`. ollama's openai-compat shim applies
   a generic chat template; it has **no dedicated Gemma-4 tool-call parser**.
   Gemma 4 emits tool calls via custom special tokens (`<|tool_call>`,
   `<tool_call|>`) and a reasoning channel. Without a matching parser those tokens
   **leak into chat output** — exactly the `<|channel|>thought<channel|>` string a
   live probe produced today. That is the "dumb" feeling: it's a **full-stack
   protocol mismatch, not model IQ** (industry framing: "do not treat tool calling
   as model-quality-only — it is a full-stack protocol problem").
2. **Zero usage reporting** on main turns (baseline: input/output both 0) — the
   openai-compat shim doesn't surface token counts, so we're flying blind on cost
   and prompt growth. (The native ollama `/api/chat` path **does** return
   `PromptEvalCount`/`EvalCount` — bridle's `provider/ollama` already maps them;
   NEX-589.)
3. **Serialization under convene.** `OLLAMA_NUM_PARALLEL=2` means a roundtable
   that wakes 4–5 nappers will queue them behind each other; latency stacks.

Points 1–2 are *fixable on ollama* by moving the keel/aspect binding from the
openai-compat lane to bridle's **native `/api/chat`** lane (already built, NEX-562/
563) — which gets us Gemma-4 template handling closer to ollama's own, plus usage
tokens, plus per-aspect `num_ctx`. Point 3 is **structural to ollama** and is the
real reason to keep vLLM on the table for the concurrent lane.

---

## Recommendation & rationale (weighting tool-calling + forge-coexistence top)

**Phase A — own the contract on ollama (do now, low risk, high payoff):**
This is the single highest-leverage move and it's mostly already built.

1. **Move keel (and other gemma aspects) off openai-compat onto bridle's native
   `provider/ollama` `/api/chat` lane.** Buys: ollama's own Gemma-4 template
   (less leakage than the generic openai shim), **real usage tokens**, per-aspect
   `num_ctx` right-sizing. The binding was intentionally left untouched in the
   timing baseline pass — flip it now and re-measure.
2. **Close the usage-reporting gap (NEX-589)** — verify `/api/chat` path reports
   tokens end-to-end into TurnFrame.Timing; the openai-compat zero-usage bug
   simply disappears on this lane.
3. **Right-size `num_ctx` per aspect** instead of the 131072 server default —
   shrinks prefill (the dominant cost in our 3 s warm TTFT) and KV footprint,
   improving both latency and forge-headroom.
4. **Make `OLLAMA_KEEP_ALIVE=-1` declarative** in
   `carriedworld-cloud/clusters/dmon/gpu/gemma-ollama.yaml` (currently a live
   `kubectl set env` that the manifest doesn't reflect; manifest still says 30m).

**Phase B — measured vLLM A/B for the concurrent/roundtable lane (do next, gated):**
Don't commit GPU budget blind. Re-deploy the parked `gemma-vllm` *with tools* and
measure it against Phase-A ollama on **the metric that matters: Gemma-4 tool-call
reliability on our real ~21-tool turns**, plus concurrent-throughput under a
simulated convene.

5. Add to `gemma-4-12b-vllm.yaml`: `--enable-auto-tool-choice`,
   `--tool-call-parser gemma4`, `--chat-template <gemma4 template>`,
   and **for coexistence**: drop `--max-model-len` to `8192` (official Gemma-4
   tool recipe) and **lower `--gpu-memory-utilization`** to a forge-safe cap
   (start ~0.40 → ~9–10 GB, leaving room for voxcpm + forge bursts). Scale to 1
   only for the test window (`strategy: Recreate`, single GPU — it already guards
   against two-pod contention).
6. If the tool-call A/B favours vLLM *and* the GPU caps hold with forge:
   **route the concurrent/convened aspect lane to vLLM** behind bridle's openai
   provider (baseURL swap, no bridle code), keep ollama for always-on singletons,
   or consolidate on vLLM if coexistence proves fine. If vLLM's open leak bugs
   (#39043/#41967) bite on our turns, **stay ollama-native** and revisit when
   upstream fixes land.

**Phase C — "own the harness" without owning the engine (the NEX-581 framing):**
The direct/own-harness path (bridle talks to the model itself, our own grammar)
is **not recommended as the engine** — it re-implements vLLM's parser for no gain
and adds a maintenance burden against a fast-moving upstream. Instead, NEX-581's
"complete harness" value is in bridle **owning the contract above the engine**:

- the **tool-call schema → parse → validate → repair-or-retry** layer (so a
  leaked-token or broken-JSON tool call gets caught and retried by *bridle*, not
  silently dropped — industry-recommended "repair-or-retry layer", and it makes
  *any* engine more reliable);
- **usage/accounting** normalization across engines;
- **per-aspect context + sampling policy**;
- **engine-swappability** as a first-class bridle capability (ollama ↔ vLLM ↔
  llama.cpp chosen per-lane by config, not code).

That is the durable investment, and it's engine-agnostic — which is *exactly* why
we shouldn't over-commit to one engine before the A/B.

---

## Migration / A/B test plan

Concrete, because the vLLM pod can just be scaled up.

**Setup**
- Phase-A ollama: flip keel's binding to native `/api/chat` (bridle config), keep
  the live deploy. Baseline already exists (turn-timing-baseline.md).
- vLLM candidate: edit `gemma-4-12b-vllm.yaml` per Phase-B #5, then
  `kubectl scale deploy/gemma-vllm -n nexus --replicas=1`; wait for `/health`
  ready (first load ~min, weights cached on PVC already). Endpoint:
  `http://gemma-vllm.nexus.svc.cluster.local:8000/v1`.

**Measure (use the existing `runtime/cmd/turnprobe` harness + observe frames):**
1. **Tool-call reliability (PRIMARY):** drive N≥50 turns per engine that *force*
   tool use across the real 21-def set (not the no-tool probes the baseline ran).
   Metric: % of turns with a clean, well-formed, schema-valid tool call (no
   leaked `<|tool_call>`/channel tokens, no broken JSON, correct args). This is
   the number the whole decision hinges on — the baseline measured 0 tool
   invocations, so **we have no tool-reliability data yet for either engine.**
2. **Concurrency:** simulate a convene — fire 4–5 concurrent tool-turns; measure
   p50/p95 completion and any queueing collapse. ollama is expected to serialize;
   quantify the gap.
3. **Latency:** warm TTFT + total, short agent turns, compare to the 3.0 s
   ollama baseline.
4. **GPU coexistence:** with vLLM at the capped util, start a forge training
   burst (or a synthetic VRAM hog) and confirm neither OOMs; watch
   `nvidia-smi` free headroom. Test vLLM sleep/wake offload if caps alone are
   tight.
5. **Usage reporting:** confirm tokens populate on both lanes.

**Decision rule**
- If ollama-native tool reliability ≥ ~vLLM and convene concurrency is tolerable
  → **stay ollama** (lowest cost, best coexistence). Bank the bridle contract work.
- If vLLM tool reliability is materially better **and** capped GPU coexists with
  forge → **vLLM for the concurrent lane**, ollama for always-on, both behind
  bridle.
- If both are flaky on Gemma-4 tool calls → the **bridle repair-or-retry layer**
  (Phase C) becomes the load-bearing fix regardless of engine; prioritize it.

---

## Honest uncertainties — what would change the answer

- **No tool-call data exists yet.** The entire baseline ran 0 tool invocations.
  The headline claim ("vLLM is better at Gemma-4 tools") rests on vLLM *having* a
  dedicated parser, not on a measurement *here*. The A/B (#1 above) is mandatory
  before any engine commitment. **This is the biggest open variable.**
- **vLLM's Gemma-4 tool parser has open leak bugs** (#39043 reasoning/tool tokens
  leak into chat; #41967 MTP speculative decoding drops first tool-call args in
  streaming multi-tool). "Has a parser" ≠ "parser is correct on our turns." Could
  go either way; must be measured on *our* template/turns.
- **q4_0 (ollama) vs FP8 (vLLM) is also a quality variable**, not just an engine
  variable — different quantization may move tool-call accuracy independently of
  the parser. If we want a clean engine comparison, consider matching quant
  (run FP8 on both, or accept q4_0-vs-FP8 as part of the package deal).
- **forge's real VRAM appetite during training is unmeasured here** (perms-locked).
  The coexistence verdict assumes forge can saturate the card; confirm its actual
  steady + burst footprint before sizing vLLM's util cap.
- **vLLM `--max-model-len` is pod-fixed** — if some aspect genuinely needs large
  context, vLLM's per-aspect right-sizing story is worse than ollama's per-request
  `num_ctx`. Confirm no aspect needs >8k for tool lanes.
- **Blackwell/sm_120 is bleeding-edge** for both (no FA3; vLLM W4A16 QAT crashed
  the nightly per the manifest comment, forcing FP8). Upstream churn could shift
  the picture month-to-month.

**Bottom line for NEX-581:** invest in bridle *owning the tool-call contract +
usage + context + engine-swap* (engine-agnostic, durable). Flip keel to ollama
native `/api/chat` now (cheap win on leakage + usage). Run the tool-reliability
A/B against the already-deployed vLLM-with-tools before spending GPU budget on
vLLM as steady-state. Don't build a from-scratch direct harness — it's the wrong
altitude and re-implements what vLLM gives free.

---

### Sources (web, 2026)
- vLLM Tool Calling docs — https://docs.vllm.ai/en/latest/features/tool_calling/
- vLLM Gemma 4 recipe (`--tool-call-parser gemma4`, template, `--max-model-len 8192`) — https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- vLLM + Gemma 4 + claude code tool-call leakage (open) — https://github.com/vllm-project/vllm/issues/39043
- Gemma4 + MTP speculative decoding drops tool-call args (open) — https://github.com/vllm-project/vllm/issues/41967
- vLLM v1 non-reclaimable KV pre-allocation — https://discuss.vllm.ai/t/vllm-v1-forces-me-to-pre-allocate-a-huge-non-reclaimable-gpu-kv-cache-for-long-contexts/1502
- vLLM sleep/wake KV offload — https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html
- ollama vs vLLM concurrency/throughput — https://tech-insider.org/vllm-vs-ollama-2026/ ; https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking
- Gemma-4 tool-call reliability (6.6%→86.4%) + ollama streaming-bug field-routing — https://gemma4home.pro/blog/gemma-4-tool-calling-troubleshooting-ollama-vllm-opencode ; https://www.morphllm.com/best-ollama-models
- Markaicode / particula engine comparisons — https://markaicode.com/vs/ollama-vs-vllm/ ; https://particula.tech/blog/ollama-vs-vllm-comparison
