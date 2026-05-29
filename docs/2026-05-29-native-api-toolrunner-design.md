# Native-API tool calling: the funnel's Go `ToolRunner`

**Date:** 2026-05-29
**Status:** Design — approved direction. Pre-implementation-plan. Open knobs in §13.
**Related:** `docs/2026-05-01-bridle-spec.md` (the harness contract this builds on); the nexus hosting design (per-aspect OS users), casket (identity), cairn (aspect identity/accountability), `CarriedWorldUniverse/lynxai` (web fetch), `feedback`/the thinking-block regression (motivation).

---

## 1. Goal & framing

Give funnel aspects a **full coding-agent tool suite on the native-API (`CategoryDirectAPI`) path**, implemented as an **all-Go `ToolRunner`** the funnel supplies to bridle. This completes the path bridle already anticipates ("the funnel supplies the implementation; the harness never owns tools") and lets aspects run on the native Anthropic/OpenAI API providers instead of the `claudecode` subprocess.

Two drivers:
- **Control + stack coherence:** in-process, all-Go, single static binary per aspect — fits the per-aspect-OS-user deployment. (We evaluated adopting `earendil-works/pi` as a generic harness; rejected — it's TypeScript/npm, so it can't be a Go `ToolRunner`, and a Node runtime per aspect cuts against the Go-single-binary stack. Its only fit would be as a subprocess provider, which is *not* the native-API direction.)
- **Durability:** the native Anthropic API provider replays thinking faithfully and sidesteps Claude Code's history-reconstruction thinking-block brick (resume/compact/auto-compact on 2.1.147+). Moving thinking-mode aspects off the `claudecode` subprocess is the durable fix.

## 2. Scope / non-goals

- **In scope:** the funnel-supplied **composite `ToolRunner`** (three lanes, §4–§7), the **autonomous permission model** (§8), identity/cred flow (§9), and lynxai-backed web access (§10).
- **Not in scope / unchanged:** bridle itself. It already owns the direct-api **tool loop** (`run.go`, `MaxSteps`), native **MCP** (`MCPClientConfig`), and **before/after-tool-call hooks**. We add no harness machinery.
- **Not building:** an agent framework, Pi adoption, or exposing the funnel *as* an MCP server.

## 3. Background — what bridle already provides

From `provider.go` / `tool.go`:
- `ProviderCategory`: **`CategoryDirectAPI`** ("bridle owns the tool loop") vs `CategorySubprocessStream` (subprocess owns tools, e.g. `claudecode`).
- `ToolDef` is **schema-only** (`Name`, `Description`, `InputSchema`) — no executor.
- `ToolRunner interface { Run(ctx, ToolCall) (json.RawMessage, error) }` — **"the funnel supplies the implementation; the harness never owns tools."**
- `ProviderCapabilities`: `SupportsCustomTools`, `SupportsBeforeToolCall`, `SupportsAfterToolCall`, `SupportsMCP`.

So this design lives entirely in the funnel's `ToolRunner` + hook implementations; bridle drives the loop and calls `Run`.

## 4. Architecture — a three-lane composite `ToolRunner`

The funnel composes one `ToolRunner` that routes by tool name into three lanes. The model sees one uniform tool surface; the runner decides local-vs-delegated.

```
            model (native API, via bridle direct-api loop)
                         │  tool_use
                         ▼
            funnel ToolRunner.Run(call)         ← BeforeToolCall hook (permissions, §8)
              ├── lane 1: LOCAL          → execute in-process (bash/file/web)         §5
              ├── lane 2: HOST/NEXUS     → delegate up to funnel→broker, identity-bound §6
              └── lane 3: EXTERNAL MCP   → bridle-native MCPClientConfig (aspect-private) §7
                         │  json.RawMessage result            ← AfterToolCall hook
                         ▼
            bridle threads result back into the loop (up to MaxSteps)
```

Compose as `LocalToolRunner` + `HostToolRunner` behind a `RouterToolRunner`; external MCP is bridle-native (not in the runner). All three are funnel-owned Go, so handlers natively hold funnel context (broker client, session, identity).

## 5. Lane 1 — local tools (execute in-runner)

The basic coding-agent set, parity-with-Claude-Code where it makes sense:
- **`bash`** — run a shell command. **Sandbox is the per-aspect OS user** (hosting design): bash runs as the aspect's user, blast radius = one homedir. No separate sandbox built. Per-aspect `cwd`, timeouts, output capture.
- **`read` / `write` / `edit`** — file read, create/overwrite, exact-string replace.
- **`glob` / `grep`** — file discovery + content search.
- **`web_fetch`** — thin HTTP client → **lynxai `POST /fetch`** → cleaned **markdown**.
- **`web_extract`** — → **lynxai `POST /extract`** (URL + JSON schema → structured JSON). A capability beyond CC's WebFetch; lynxai's cred vault also enables **logged-in** fetches.
- **`web_search`** — provider TBD (§13).

## 6. Lane 2 — host / nexus-native tools (delegate up, identity-bound)

Tools where the aspect acts **as itself in the nexus network**: `send_comms`/`send_chat`, issues/tickets, `dispatch`, file-announce (Files Subsystem `s3://`), roster/thread queries, scheduler. `Run` does **not** execute locally — it calls back up into the funnel/host (broker connection, session).

**Core principle — identity is funnel-injected, never model-supplied.** When the model calls `send_comms`/`create_issue`, the funnel stamps the **authenticated aspect identity** (from the casket/cairn identity layer) server-side. The tool's input schema carries no identity field (or the funnel overrides it). The model **cannot forge "from another aspect."** This is the accountability boundary, and it's *why* these tools are host-mediated rather than local.

**Async:** `ToolRunner.Run` is synchronous (returns `json.RawMessage`). Per delegated tool:
- **ack-now** (fire-and-forget): `send_comms` returns an ack immediately.
- **block-until-result**: a roster/thread query blocks and returns data.
- **handle + out-of-band**: `dispatch` returns a handle; the subagent's reply arrives later via comms (not as the tool result).

## 7. Lane 3 — external MCP (aspect-private, nexus-independent)

An individual aspect's own MCP server connections — third-party tools, **independent of the nexus network**. Handled by **bridle-native direct-api MCP** (`MCPClientConfig`), per-aspect, per-turn. Carries **no nexus identity** — a distinct trust domain. Creds for these servers are the aspect's own, vended from the nexus cred store but the connection is direct (bridle ↔ MCP server). (Contrast lane 2's centralized, host-owned, credentialed nexus services.)

## 8. Permission model (autonomous)

Claude Code prompts a human on risky tools; **aspects can't be prompted**. So permissions are a **per-aspect config policy enforced at bridle's `BeforeToolCall` hook**:
- Each aspect has an allow/deny policy (per tool, optionally per-arg-pattern — e.g. `bash` allowlist, write-paths).
- A denied call → either a tool error the model can react to, or **escalation to the operator** (via the comms/operator layer) for a human decision, then resume.
- Default postures by lane: **local** (esp. `bash`/`write`) = guarded by policy; **host/nexus-native** = identity-bound already, lighter; **external MCP** = per-aspect allowlist of servers/tools.

`AfterToolCall` hook = observability (log tool + result for the aspect's audit trail).

## 9. Identity & credentials

- **Identity** flows from the casket (Ed25519 channel) / cairn (aspect git identity) layer; the funnel injects it on lane-2 tools (§6). One identity, surfaced in the tool layer.
- **Credentials** (provider API keys, lynxai endpoint+key, external-MCP creds) are **vended on demand from the nexus credential store** (the NEX-336 broker-resolved-creds pattern) — not env/script literals. Lane-2 host services use host-owned creds (centralized/audited); lane-3 external MCP uses the aspect's own creds.

## 10. lynxai integration (web access)

`lynxai` = self-hosted AI-native headless browser (Go, Chromium/chromedp), HTTP server, AGPL. `web_fetch` → `/fetch` (markdown); `web_extract` → `/extract` (schema → JSON). Its **encrypted credential vault** enables logged-in fetches, and a future **`/drive`** endpoint (agent drives a browser UI, e.g. to bootstrap an API key) could become a `web_drive` tool later.
- **Deployment:** a **shared, self-hosted lynxai service**; the funnel's web tools are thin HTTP clients; endpoint+key vended from nexus. Shared-vs-per-aspect and `/drive`-as-a-tool are open (§13).

## 11. Deployment

- **funnel + bridle + ToolRunner = one static Go binary per aspect** — fits the per-aspect-OS-user model (hosting design); bash runs as that user = built-in sandbox.
- **lynxai = a separate self-hosted HTTP service** (its own cred vault), shared across aspects.
- Native-API aspects use bridle's `claude`/`openai` providers + this `ToolRunner`; they can **coexist** with subprocess-provider aspects during migration (§14).

## 12. Error handling & testing

- **Errors:** tool failures return as tool_result errors so the model can see and react; permission-deny → error or operator-escalation; lynxai/MCP unreachable → tool error (not a crash). Per-aspect-user means a bad `bash` can't escape the homedir.
- **Testing:** bridle already ships `fake/tool_runner.go` — use it. Per-lane unit tests (local exec; host-delegate with a fake broker; identity-injection asserts the model can't override identity); permission-policy tests; lynxai client against a fake `/fetch`/`/extract` server.

## 13. Open knobs

- **`web_search` provider** (dedicated search API vs lynxai-extract over a SERP).
- **Exact local-tool list / parity** with Claude Code (which tools, names, schemas).
- **lynxai:** shared-vs-per-aspect; expose `/drive` as `web_drive`.
- **Async-delegated-tool conventions** (the ack / block / handle taxonomy, §6) — formalize per tool.
- **Which aspects migrate** off `claudecode` first (thinking-mode aspects benefit most).

## 14. Migration

Aspects move from the `claudecode` subprocess provider → a native API provider (`claude`/`openai`) + this `ToolRunner`. This is the durable thinking-block-brick fix. Migration is per-aspect and incremental — subprocess and native aspects coexist. Thinking-heavy aspects should go first.

## 15. Sequencing (rough)

- **P1** — `LocalToolRunner`: bash (per-user) + file + glob/grep + `web_fetch`/`web_extract` (lynxai). Native-API turn loop end-to-end with a minimal tool set.
- **P2** — `HostToolRunner`: `send_comms` + issues, with **funnel identity injection**; the `RouterToolRunner`.
- **P3** — permission model (`BeforeToolCall` policy + operator escalation) + `AfterToolCall` audit.
- **P4** — external MCP wiring (lane 3) + cred vending from nexus.
- **P5** — `dispatch` + the async-delegated taxonomy + remaining host tools; `web_search`.

Each phase its own implementation plan.

## 16. Operational safety (build sessions)

Per the thinking-block bug: run build sessions on **pinned CC 2.1.146 or `MAX_THINKING_TOKENS=0`** (zero thinking blocks), MCP avoided in thinking-on sessions, committed per step.
