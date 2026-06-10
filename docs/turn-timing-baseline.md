# Turn-timing baseline — 2026-06-10

Phase-2 measurement pass of the bridle optimization plan
(`docs/superpowers/plans/2026-06-10-bridle-optimization.md`). Captured on the
dMon k3s cluster immediately after deploying nexus main `e7ccbca`
(NEX-563 / PR #354: TurnFrame.Timing forwarded from bridle TurnTiming) into
the broker + builder/shadow/maren images, and after the gemma server-side
keep-alive fix (`OLLAMA_KEEP_ALIVE=-1`).

Method: an operator WS client (throwaway `runtime/cmd/turnprobe` on
little-blue, auth-bypass connect) sent 10 short DM probes per lane
("timing probe N: reply with one sentence") via `aspect.say`, subscribed to
`subscribe.observe`, and recorded every completed `observe.frame` turn
snapshot. Raw JSONL on little-blue at `/tmp/turnbaseline/{keel,shadow}.jsonl`.
All values are seconds from the new `timing` block (NEX-561 instrumentation,
injected-clock spans at the harness seam).

## Per-lane table

| lane | n | total med | total p90 | startup→TTFT med | startup→TTFT p90 | stream med | tools | prompt bytes med | msgs med | tokens (med) |
|---|---|---|---|---|---|---|---|---|---|---|
| keel main — gemma 11.9B, ollama via openai-compat, always-on pod (all turns) | 10 | 5.80 | 15.97 | 3.19 | 14.90 | 0.09 | 0 | 2 056 | 9 | not reported (0/0) |
| keel main — warm only (turns 2–10) | 9 | 5.59 | 7.91 | 3.03 | 5.07 | 0.10 | 0 | 2 305 | 10 | not reported (0/0) |
| shadow main — claude-code CLI lane, always-on cloud pod | 10 | 3.50 | 5.20 | 2.65 | 4.48 | 0.72 | 0 | 442 | 2 | in 5 / out 9 / cache_read 39 368 |
| filter-judge — deepseek-v4-flash (cloud), keel+shadow combined | 14 | 2.74 | 3.14 | 2.59 | 3.17 | 0.18 | 0 | 281 | 1 | in 813 / out 133 |
| anvil builder — codex CLI (dispatch Job) | 0 | — | — | — | — | — | — | — | — | BLOCKED (see below) |

Notes per lane:

- **keel**: `tool_def_count=21` on every round. Prompt grows with the
  global-context session (240 B / 1 msg on turn 1 → 5 116 B / 22 msgs by
  turn 10). Stream time is negligible (≤0.33 s) — virtually the whole warm
  turn is startup→TTFT, i.e. gemma prefill/queue on the ollama server.
  The openai-compat lane reports **no usage tokens** on main turns
  (input/output both 0; the judge lane on the same provider does report) —
  instrumentation gap worth a ticket.
- **shadow**: the CLI lane sends only the last user message
  (prompt ~442 B, 1–2 msgs) and resumes the CLI's own session
  (cache_read ~39 k tokens). startup→TTFT med 2.65 s **is the
  claude-code spawn + session-resume + TTFT cost — the headline CLI-lane
  number.** Total med 3.50 s, worst 5.61 s. All 10 turns completed and
  replied but were marked `status=errored`: claude-code prints a stderr
  warning ("Claude configuration file not found at /root/.claude.json, a
  backup exists...") which the claudecode provider classifies as a turn
  error. Cosmetic but pollutes the status field — pre-existing, not from
  this deploy; worth a ticket (restore `.claude.json` in the shadow pod
  and/or whitelist that stderr pattern).
- **filter-judge**: runs after every main turn and gates posting, so its
  ~2.7 s is added to every user-visible reply on every lane.

## keel keep-alive: before/after

- Declarative source: `carriedworld-cloud/clusters/dmon/gpu/gemma-ollama.yaml`
  had `OLLAMA_KEEP_ALIVE=30m` — model unloads after 30 min idle, so every
  re-engagement after a quiet spell paid a full model load.
- Observed cold turn (#1, model not resident): **total 70.50 s, of which
  70.47 s startup→TTFT** — ~65 s of pure model load vs the 2–10 s warm band.
- Fix applied live: `kubectl set env deployment/gemma-ollama -n nexus
  OLLAMA_KEEP_ALIVE=-1` (box is dedicated). After restart + first turn,
  `/api/ps` shows gemma pinned with `expires_at: 2318-09-20...` (never).
  Warm turns 2–10 ran 2.16–9.91 s.
- The hosting-reconcile CronJob applies `clusters/dmon/` **non-recursively**
  (`hosting/apply.sh`), so the live env change will NOT be reverted — but the
  manifest in `clusters/dmon/gpu/gemma-ollama.yaml` still says `30m` and
  should be updated in carriedworld-cloud to make the fix declarative.

## Builder (codex) lane — blocked

Two dispatch attempts (`!dispatch anvil ...`, trivial probe brief; second
attempt with `repo=CarriedWorldUniverse/bridle` so codex had a worktree).
Job spin-up worked well (job created → funnel validated ≈ 2.6 s; clone/
worktree + gh bridge ≈ +2.2 s), but every codex subprocess exited status 1
in ~19 s. Reproduced in-pod: `codex exec` gets **401 Unauthorized
("Missing bearer or basic authentication") from wss://api.openai.com/v1/responses**
— the `codex-auth` k8s secret's auth.json is expired/invalid (secret is 5
days old). Needs an interactive codex re-login to refresh; not fixable from
this session. Both probe jobs deleted; no branches/PRs were created.
Secondary observation: no observe.frame ever arrived for the failed builder
turns — the errored-turn path from a builder goal-loop `Pursue` error
apparently never emits a terminal turn snapshot; worth checking when the
codex auth is fixed. Also: with no `repo=`, agentfunnel never sets bridle's
`SkipGitRepoCheck`, so a repo-less dispatch cannot work on the codex lane at
all (first attempt's failure mode).

## Phase-3 decision input — where the time goes (ranked)

1. **Gemma cold model load (was ~65–70 s per cold engagement) — FIXED** by
   the deployed `OLLAMA_KEEP_ALIVE=-1`. Structurally gone while the env var
   sticks; make it declarative in carriedworld-cloud.
2. **Filter-judge adds ~2.7 s (med) to every posted reply on every lane.**
   It is the single largest *remaining* fixed cost: ~50% of a warm keel turn
   and ~75% of a shadow turn, again, after the turn. Options, in measured
   order of payoff: route the judge to the now-pinned local gemma (zero
   cloud RTT; keel judge turns prove the openai lane shape already works),
   skip the judge for trivial/DM-ack turns (hard rules already exist), or
   overlap judge with reply delivery.
3. **keel warm startup→TTFT ~3.0 s med / 5.1 s p90** — gemma queue+prefill.
   Prompt is small (≤5 KB, 21 tool defs), so this is server-side: 131 k
   `OLLAMA_CONTEXT_LENGTH` with q8_0 KV and 2-way parallel. The new native
   ollama lane (`OLLAMA_BASE_URL` + `OLLAMA_NUM_CTX` per-aspect num_ctx)
   can right-size the context per aspect — worth an A/B once keel's binding
   moves off openai-compat (binding intentionally untouched in this pass).
4. **claude-code CLI spawn+resume+TTFT 2.65 s med / 4.48 s p90** — the CLI
   lane's per-turn floor. Bounded and tolerable for DM traffic; the
   measurement-gated subprocess work (PR-4 consolidation, keep-warm ideas)
   should be judged against this ~2.6 s, not against keel's numbers.
5. **Assembly + stream are noise** — assembly is µs; stream ≤0.93 s
   everywhere (one-sentence replies). Tool time was 0 in this probe set
   (no tools invoked); re-measure tool-heavy turns when a builder lane works.

Instrumentation gaps found while measuring: (a) openai-compat main turns
report zero usage tokens; (b) builder goal-loop error path emits no terminal
observe frame; (c) shadow's stderr-noise → `errored` status mislabels every
healthy turn.

## Appendix — raw per-turn JSON (events elided)

keel probe 1 (cold load):

```json
{"turn_id": "turn-20260610T101243.037201Z-46a542", "label": "main", "status": "complete", "started": "2026-06-10T10:12:43.051317371Z", "ended": "2026-06-10T10:13:53.559925563Z", "trigger_msg": 7960, "model": "gemma", "provider": "openai", "usage": {"input_tokens": 0, "output_tokens": 0, "duration_ns": 70508608211}, "timing": {"rounds": [{"assembly_secs": 1.3181e-05, "startup_to_first_event_secs": 70.466783794, "stream_secs": 0.03582675, "prompt_bytes": 240, "message_count": 1, "tool_def_count": 21}], "total_secs": 70.502727934}}
```

keel probe 6 (warm):

```json
{"turn_id": "turn-20260610T101430.448261Z-5acb58", "label": "main", "status": "complete", "started": "2026-06-10T10:14:30.477937151Z", "ended": "2026-06-10T10:14:32.632352412Z", "trigger_msg": 7971, "model": "gemma", "provider": "openai", "usage": {"input_tokens": 0, "output_tokens": 0, "duration_ns": 2154415232}, "timing": {"rounds": [{"assembly_secs": 8.2473e-05, "startup_to_first_event_secs": 2.120126614, "stream_secs": 0.037996467, "prompt_bytes": 2305, "message_count": 10, "tool_def_count": 21}], "total_secs": 2.158262344}}
```

shadow probe 1 (claude-code CLI; note the cosmetic `errored` from stderr noise):

```json
{"turn_id": "turn-20260610T101541.208157Z-10da78", "label": "main", "status": "errored", "started": "2026-06-10T10:15:41.22292642Z", "ended": "2026-06-10T10:15:44.900245539Z", "trigger_msg": 7985, "model": "claude-opus-4-7", "provider": "claude-code", "usage": {"input_tokens": 5, "output_tokens": 9, "cache_create": 39158, "duration_ns": 3677319138}, "timing": {"rounds": [{"assembly_secs": 1.5219e-05, "startup_to_first_event_secs": 2.899952918, "stream_secs": 0.771958726, "prompt_bytes": 242, "message_count": 1}], "total_secs": 3.672029887}, "error": "claudecode: stderr output: Claude configuration file not found at: /root/.claude.json ..."}
```

filter-judge (keel turn 1's judge):

```json
{"turn_id": "turn-20260610T101243.037201Z-46a542-judge", "label": "filter-judge", "status": "complete", "started": "2026-06-10T10:13:53.560154809Z", "ended": "2026-06-10T10:13:56.412320605Z", "trigger_msg": 7960, "model": "deepseek-v4-flash", "provider": "openai", "usage": {"input_tokens": 813, "output_tokens": 110, "duration_ns": 2852165778}, "timing": {"rounds": [{"assembly_secs": 1.94e-05, "startup_to_first_event_secs": 2.707506241, "stream_secs": 0.163708279, "prompt_bytes": 259, "message_count": 1}], "total_secs": 2.871241152}}
```

Per-turn series (totals, seconds):

- keel total: 70.5, 9.9, 5.4, 4.6, 4.5, 2.2, 7.4, 5.6, 7.2, 6.0
- keel startup→TTFT: 70.5, 8.7, 4.2, 3.4, 3.0, 2.1, 2.8, 2.8, 3.3, 2.7
- shadow total: 3.7, 5.1, 2.6, 4.2, 2.9, 2.8, 5.6, 3.3, 2.5, 4.3
- shadow startup→TTFT: 2.9, 4.4, 1.9, 3.5, 2.1, 2.1, 4.9, 2.4, 1.8, 3.6

Full raw captures: little-blue `/tmp/turnbaseline/keel.jsonl` (26 frames incl.
judge + filter-decision) and `/tmp/turnbaseline/shadow.jsonl` (22 frames).
