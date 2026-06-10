# Bridle Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-turn timing + request-size instrumentation in bridle, ollama keep_alive/num_ctx exposure, nexus forwarding, a live baseline on dMon, then mechanical consolidation of duplicated subprocess plumbing.

**Architecture:** Spec at `docs/superpowers/specs/2026-06-10-bridle-optimization-design.md`. All instrumentation lands at the harness seam (the one `h.provider.RunTurn(ctx, preq, sink)` site in run.go and the `sink.Emit` call sites around it) via a stamping/recording sink decorator — zero provider changes. Measurement-gated perf fixes are explicitly NOT in this plan; they get planned after the baseline table exists.

**Tech Stack:** Go; bridle (this repo), nexus (PR 3), ollama Go client (`api.ChatRequest`).

---

## PR 1 — Instrumentation (branch `feat/turn-timing`)

### Task 1: TurnTiming types + injected clock + total/assembly timing

**Files:**
- Modify: `harness.go` (TurnResult ~line 211, Harness ~line 227)
- Modify: `run.go` (runTurn entry, lowerRequest call)
- Test: `run_timing_test.go` (new; follow the fake-provider patterns in `harness_test.go` / `fake/`)

- [ ] **Step 1: Failing test** — fake provider + fake clock; assert TotalSecs and the first round's AssemblySecs are computed from the injected clock:

```go
func TestTurnTimingTotalAndAssembly(t *testing.T) {
	clk := &fakeClock{t0: time.Unix(1000, 0)} // Now() advances 100ms per call; helper in this test file
	h := NewHarness(&fake.Provider{...})      // single text round, no tools (use the existing fake)
	h.now = clk.Now
	res, err := h.RunTurn(context.Background(), basicReq(), nil, &fake.SliceEventSink{})
	// assertions: res.Timing.TotalSecs > 0; len(res.Timing.Rounds) == 1;
	// res.Timing.Rounds[0].AssemblySecs > 0; values consistent with the scripted clock steps
}
```

(The fake clock must be deterministic: a step counter, not wall time. Exact expected values follow from how many Now() calls the implementation makes — pin them once implemented, asserting exact equality, not just >0, so regressions in call placement are caught.)

- [ ] **Step 2: Run** `go test -run TestTurnTiming ./ -v` — FAIL (no Timing field / no now field).

- [ ] **Step 3: Implement.** In `harness.go`:

```go
// RoundTiming captures where one provider round spent its time and
// what it sent. Secs floats (not Durations) so the struct marshals
// readably into TurnFrame JSON downstream.
type RoundTiming struct {
	AssemblySecs            float64 // request assembly + BeforeModelCall hooks
	StartupToFirstEventSecs float64 // provider call -> first sink event (CLI lane: spawn+startup+TTFT)
	StreamSecs              float64 // first event -> provider call return
	PromptBytes             int     // marshaled request messages size
	MessageCount            int
	ToolDefCount            int
}

// ToolTiming is one tool call's wall-clock duration.
type ToolTiming struct {
	ID   string
	Name string
	Secs float64
}

// TurnTiming aggregates per-turn instrumentation. Zero value = not recorded.
type TurnTiming struct {
	Rounds    []RoundTiming
	Tools     []ToolTiming
	TotalSecs float64
}
```

Add `Timing TurnTiming` to `TurnResult`. Add to `Harness`:

```go
type Harness struct {
	provider Provider
	hooks    hookRegistry
	now      func() time.Time // injectable clock; nil means time.Now
}
```

with a `func (h *Harness) clock() func() time.Time` accessor returning `time.Now` when nil. In `run.go` runTurn: capture `turnStart := now()`; bracket the assembly span (lowerRequest + BeforeModelCall) per round; set `result.Timing.TotalSecs` just before the TurnDone emit.

- [ ] **Step 4:** Test passes; full `go test ./... -count=1` green.
- [ ] **Step 5: Commit** `feat: TurnTiming on TurnResult — total + assembly spans (injected clock)`

### Task 2: Stamping sink decorator — event timestamps + startup/stream split

**Files:**
- Modify: `events.go` (add TS fields), `run.go` (wrap sink once at runTurn entry)
- Create: `timing_sink.go`
- Test: `run_timing_test.go` (extend), `timing_sink_test.go`

- [ ] **Step 1: Failing tests:** (a) every event received by the caller's sink carries a non-zero TS from the injected clock; (b) with a fake provider that emits two ModelChunks, `Rounds[0].StartupToFirstEventSecs` covers provider-call→first chunk and `StreamSecs` covers first chunk→return (pin exact fake-clock values).

- [ ] **Step 2: Implement.** Additive `TS time.Time` on `ModelChunk`, `ToolCallStart`, `ToolCallResult`, `StepBoundary`, `TurnDone`, `TurnError` in events.go. New `timing_sink.go`:

```go
// stampSink decorates the caller's EventSink: stamps TS on every event
// and records the first-event time per provider round for TurnTiming.
// The harness wraps the caller's sink once at runTurn entry and passes
// the wrapped sink to the provider, so provider implementations never
// change.
type stampSink struct {
	inner EventSink
	now   func() time.Time

	mu         sync.Mutex
	firstEvent time.Time // zero until the round's first event; reset per round
}

func (s *stampSink) Emit(ev Event) {
	ts := s.now()
	s.mu.Lock()
	if s.firstEvent.IsZero() {
		s.firstEvent = ts
	}
	s.mu.Unlock()
	s.inner.Emit(stampEvent(ev, ts)) // type switch setting the TS field
}

// roundReset clears the first-event mark; called by runTurn before each
// provider round. takeFirstEvent returns and clears it after the round.
```

run.go: wrap once (`sink = &stampSink{inner: sink, now: h.clock()}`), call roundReset before `h.provider.RunTurn`, compute the two spans after it returns.

- [ ] **Step 3:** Tests pass (`-race` too); commit `feat: event timestamps + per-round startup/stream timing via sink decorator`.

### Task 3: Tool durations + request-size accounting

**Files:**
- Modify: `run.go` (executeToolCall brackets ~lines 255–314; round setup)
- Test: `run_timing_test.go` (extend)

- [ ] **Step 1: Failing tests:** (a) a fake-provider turn with one tool call yields `Timing.Tools` = [{id, name, secs from fake clock}]; (b) `Rounds[0].PromptBytes` equals `len(json.Marshal(preq.Messages))` for the assembled request and MessageCount/ToolDefCount match.
- [ ] **Step 2: Implement:** bracket executeToolCall with the clock (the ToolCallStart/Result emit sites already delimit it); append ToolTiming. At round setup, after lowerRequest: marshal `preq.Messages` for PromptBytes (on marshal error record -1, never fail the turn), set MessageCount=len(preq.Messages), ToolDefCount=len(preq.Tools).
- [ ] **Step 3:** Tests pass; full suite + vet green; commit `feat: per-tool durations + request size in TurnTiming`. Open PR 1.

## PR 2 — Ollama options (branch `feat/ollama-options`)

### Task 4: KeepAlive / NumCtx / Options passthrough

**Files:**
- Modify: `provider/ollama/ollama.go` (Provider struct ~line 21, ChatRequest build ~line 77)
- Test: `provider/ollama/ollama_test.go` (create if absent — there are currently NO tests in this package; use httptest against the ollama JSON API shape, mirroring how provider/openai tests fake their endpoint)

- [ ] **Step 1: Failing tests:** build a Provider with `KeepAlive: 45*time.Minute, NumCtx: 8192, Options: map[string]any{"temperature": 0.2}`; run a turn against an httptest server that records the request body; assert the body has `"keep_alive":"45m0s"` (or the client's serialization — pin what the ollama client actually sends), `"options":{"num_ctx":8192,"temperature":0.2}`. Second test: zero-value Provider sends keep_alive `"30m0s"` (the spec default) and NO num_ctx key.
- [ ] **Step 2: Implement:**

```go
type Provider struct {
	client    *api.Client
	baseURL   string
	KeepAlive time.Duration  // 0 means the 30m default (spec: keel is always-on)
	NumCtx    int            // 0 means omit (model default)
	Options   map[string]any // merged into ChatRequest.Options; NumCtx wins on conflict
}
```

In the ChatRequest build: start Options from a copy of p.Options (never mutate the field), set `options["num_ctx"] = p.NumCtx` when >0; `ka := p.KeepAlive; if ka == 0 { ka = 30 * time.Minute }; chatReq.KeepAlive = &api.Duration{Duration: ka}` (verify the field name/type on the vendored ollama client first — adjust to its actual API).

- [ ] **Step 3:** Tests pass; suite green; commit `feat(ollama): keep_alive + num_ctx + options passthrough`. Open PR 2.

## PR 3 — nexus forwarding (branch `feat/turn-timing-forward`, repo ~/Source/nexus)

### Task 5: Bump bridle; TurnFrame.Timing; per-aspect ollama options config

**Files (nexus repo):**
- Modify: `go.mod` (bridle bump to the PR-1/PR-2 merge), `nexus/observability/types.go` (TurnFrame), the funnel's TurnDone→TurnFrame path (find via `grep -rn "TurnDone" runtime/ nexus/` — the Grouper that builds TurnFrames), and the per-aspect provider-construction site for ollama (find via `grep -rn "ollama.New\|ollama.Provider" --include=*.go`).
- Test: alongside the touched files, following their existing test patterns.

- [ ] **Step 1:** `go get github.com/CarriedWorldUniverse/bridle@<merged-sha>` (pseudo-version per the pinning policy: code-source builds track main HEAD).
- [ ] **Step 2: Failing test:** the Grouper's TurnFrame for a completed turn carries `Timing` (marshaled from bridle's TurnTiming) when the TurnDone result has it.
- [ ] **Step 3: Implement:** additive `Timing *bridle.TurnTiming \`json:"timing,omitempty"\`` on TurnFrame (or a mirrored local struct if observability must not import bridle — follow whatever the file already does for Usage, which mirrors rather than imports; if mirroring, mirror exactly and convert). Fill it where TurnDone is consumed.
- [ ] **Step 4: ollama config:** thread `keep_alive`/`num_ctx` from the per-aspect provider config (the admin_model_config / provider-binding seam — read `nexus/broker/admin_provider_binding.go` and where keel's provider is constructed) into `ollama.Provider{KeepAlive, NumCtx}`. Config keys additive with safe zero-values. Test at the construction site.
- [ ] **Step 5:** Full nexus suite green; commit; open PR 3. Trace pane note: agora's TurnFrame mirror tolerates unknown JSON fields (it decodes selectively), so no agora change is required for the timing field to flow; rendering timing in the pane is a later, separate nicety.

## Phase 2 — Measurement pass (operational, after PRs 1–3 deploy)

Not a coding task. Procedure:

1. Deploy: rebuild broker + keel/cloud-shadow funnel images on dMon (per `deploy/broker/build.sh` convention + the aspect image equivalents), restart pods.
2. Drive ~10 representative DM turns each: keel (gemma/ollama), cloud-shadow (claudecode), one builder turn (codexcli), maren if convenient.
3. Pull TurnFrames (observe stream or broker logs) and tabulate: per lane — TotalSecs, StartupToFirstEventSecs, StreamSecs, tool time, PromptBytes, tokens in/out.
4. Commit `docs/turn-timing-baseline.md` (dated header) to bridle with the table + the Phase-3 decision (which measured cost gets fixed, with numbers). Measurement-gated fixes get their own plan from that doc.

## PR 4 — Consolidation (branch `refactor/internal-subprocess`, AFTER the baseline is committed)

### Task 6: Extract `internal/subprocess` helpers

**Files:**
- Create: `internal/subprocess/subprocess.go`, `internal/subprocess/subprocess_test.go`
- Modify: `provider/claudecode/claudecode.go`, `provider/codexcli/codexcli.go`, `provider/geminicli/geminicli.go` (delete the duplicated copies, call the package)

- [ ] **Step 1:** Move byte-identical pieces, one commit per extraction so each is independently revertable:
  1. cancel watcher (SIGTERM → 5s grace → SIGKILL; identical at claudecode:240–256, codexcli:120–134, geminicli:143–159) → `subprocess.WatchCancel(ctx, cmd, procExited)`;
  2. `mergeEnv` (claudecode:729–751 ≡ codexcli:514–535) → `subprocess.MergeEnv(base, overlay)`;
  3. `buildPrompt` last-user-message extraction (claudecode:716–723 ≡ codexcli:505–512) → `subprocess.LastUserPrompt(msgs)` (antigravitycli keeps its full-context variant);
  4. JSONL scanner skeleton (1MB buffer + TrimSpace + skip-empty + per-line callback) → `subprocess.ScanJSONLines(r io.Reader, perLine func([]byte) error)`; each provider keeps its own event-schema dispatch inside its callback;
  5. error-classification table type (claudecode:800–866's `classificationPattern` struct + lookup loop) → `subprocess.Classify(stderr string, patterns []subprocess.Pattern) ...`; the pattern lists stay in each provider.
- [ ] **Step 2 (per extraction):** unit-test the extracted helper directly in `subprocess_test.go` (the watcher with a fake command; MergeEnv/LastUserPrompt/ScanJSONLines as pure functions); then the acceptance gate: the three providers' EXISTING suites pass byte-unchanged (`go test ./provider/... -count=1`). No behavior change permitted — if a provider's copy turns out to differ subtly, leave that provider's copy in place and note it in the commit rather than "harmonizing".
- [ ] **Step 3:** Full suite + vet + gofmt; open PR 4.

## Self-review notes

- Spec coverage: Phase 1 → Tasks 1–3; Phase 3 no-regret → Task 4 (+ Task 5 config); nexus forwarding → Task 5; Phase 2 → procedure section; Phase 4 → Task 6. Measurement-gated fixes intentionally absent (spec: planned post-baseline).
- Exact fake-clock expected values are pinned at implementation time (Step-1 notes say assert exact equality once call placement is known) — deliberate, not a placeholder: the call count is an implementation property the test locks in.
