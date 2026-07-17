# bridle-claude-sidecar

Thin, disposable Node process wrapping `@anthropic-ai/claude-agent-sdk`
over stdio JSON-lines, spoken to by bridle's Go `provider/claudesdk`
(NEX-745). See `~/bridle-agentsdk-spec.md` §2/§5 for the architecture
rationale (sidecar over a hand-rolled `--input-format stream-json`
control protocol — P5: buy the wire protocol from the vendor, don't own
an undocumented one).

Not an independently-versioned/published package — it lives inside the
`bridle` repo, builds with `npm`, and is meant to be rebuilt whenever
`provider/claudesdk`'s wire protocol version changes (`wire.go`'s
`protocolVersion` / `wire.ts`'s `PROTOCOL_VERSION` must match).

## Build

```sh
cd bridle-claude-sidecar
npm install
npm run build      # tsc -> dist/index.js (+ jsonSchema.js, wire.js)
```

`npm run typecheck` runs `tsc --noEmit` only, for CI-style checks
without producing `dist/`.

## Packaging decision (spec §11)

**Image-bundled, not `npx`-resolved.** The worker image that carries
this lane bundles `bridle-claude-sidecar/dist/` (built at image-build
time) next to the `claude` CLI binary it already ships for `claudecode`
— no network fetch at turn-spawn time, matching the operator's stated
preference and the same trust/packaging model as the `claude` binary
itself. `provider/claudesdk.Provider.SidecarPath` defaults to
`"bridle-claude-sidecar"` (a PATH-resolved wrapper script the image
installs, e.g. `#!/bin/sh\nexec node /opt/bridle/bridle-claude-sidecar/dist/index.js "$@"`),
mirroring how `claudecode.Provider.ClaudePath` defaults to `"claude"`.

This repo does not build the worker image — that wiring is a separate
deployment concern once this lane lands (spec §9: "The OpenAI-compatible
HTTP front itself — separate thin unit"). `dist/` and `node_modules/`
are gitignored; the image build step is expected to run `npm ci && npm
run build` from this directory.

## Wire protocol

Private to the bridle↔sidecar pair — see `src/wire.ts` (TypeScript side)
and `../provider/claudesdk/wire.go` (Go side). The two must be kept in
sync by hand; there is no shared schema generator for v1. Bump both
`PROTOCOL_VERSION` constants together on any breaking change — a
mismatched pair rejects loudly at turn start (`error` event, exit 1),
never mid-turn.

## Known v1 simplifications

- **JSON-Schema→Zod bridge is best-effort** (`src/jsonSchema.ts`): the
  Agent SDK's in-process MCP tool registration (`tool()`) is typed to a
  Zod raw shape, not a JSON Schema. Property *names* and required-ness
  are preserved exactly (that's what protects Claude from seeing the
  wrong argument names — the thing spec's "must not remangle" is
  actually about); coarse per-property types (string/number/boolean/
  array/object) are mapped to their closest Zod primitive; anything the
  bridge can't confidently narrow (nested schemas beyond one level,
  enums beyond string enums, patterns, `oneOf`/`anyOf`, `minLength`,
  etc.) degrades to `z.any()` rather than being dropped or rejected.
- **No token-level streaming**: each `text_delta`/`thinking_delta` event
  is emitted once per COMPLETE content block from the SDK's `assistant`
  messages, not per output token (`options.includePartialMessages` is
  not enabled). This mirrors `claudecode`'s own per-block granularity
  (`EmitAssistantText` per text content block in `claudecode.go`), not a
  new corner cut for this lane.
- **`canUseTool` always default-allows and ignores
  `before_tool_call_gate`.** The SDK's native-tool permission gate is a
  separate mechanism from bridle's own `BeforeToolCall` hook, which
  fires authoritatively on bridle's side of the custom-tool round trip
  (`provider/claudesdk`'s `ToolExecutor` path →
  `Harness.executeToolCall`). It does not currently give bridle a
  second opinion on the SDK's own native-tool permission decisions —
  `before_tool_call_gate` is sent on the wire but unread here, a
  forward-compatible placeholder for a future round trip that doesn't
  exist yet. Because of this, `Capabilities().SupportsBeforeToolCall` is
  **mode-dependent**, not a blanket `true`: `ModeFunnel` (all tool calls
  are bridle-custom, really gated) advertises `true`; `ModeAgent`
  (native tools live and ungated) advertises `false`, so a caller can't
  be misled into trusting a veto of a native call that can never
  happen.
- **Cancellation is SIGTERM-only from the Go side.** The wire protocol
  defines an in-band `interrupt` message (honoured if a caller sends one
  — see `main()`'s inbound line router) but `provider/claudesdk`'s Go
  side doesn't send it; it relies on
  `internal/subprocess.SetPgid`+`WatchCancelGroup`'s
  SIGTERM→grace→SIGKILL, signaling the WHOLE process group this sidecar
  leads (not just its own PID) so the real `claude` CLI child it spawns
  is reaped too — unlike `claudecode`, which has no such grandchild and
  keeps using plain `WatchCancel` unchanged. The sidecar itself also
  installs a SIGTERM/SIGINT handler (`installShutdownHandler` in
  `index.ts`) as a second, independent layer: it best-effort
  `interrupt()`s the live SDK query before exiting, in case this process
  ever receives the signal directly rather than solely via bridle's
  group kill.
