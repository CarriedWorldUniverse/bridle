# bridle

[![CI](https://github.com/CarriedWorldUniverse/bridle/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CarriedWorldUniverse/bridle/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CarriedWorldUniverse/bridle?include_prereleases&sort=semver&display_name=tag)](https://github.com/CarriedWorldUniverse/bridle/releases)
[![Go Reference](https://pkg.go.dev/badge/github.com/CarriedWorldUniverse/bridle.svg)](https://pkg.go.dev/github.com/CarriedWorldUniverse/bridle)
[![License](https://img.shields.io/github/license/CarriedWorldUniverse/bridle)](LICENSE)

The harness layer of the Nexus funnel/harness split. A small Go library that drives one deliberation turn of one model with one tool surface, emitting a stream of observable events and returning a structured `TurnResult`.

The funnel imports bridle. Aspects do not import bridle directly.

> A bridle controls and directs without being the horse — sits beneath the funnel, governs what the model produces, provider-agnostic.

## Status

Implemented and in active use. The original build brief is preserved in
[`docs/2026-05-01-bridle-spec.md`](docs/2026-05-01-bridle-spec.md); the
current code includes direct API providers, headless CLI providers, hook
support, MCP plumbing for direct providers, and a local tool runner.

## Scope

- One stable provider interface with direct-API and subprocess-stream implementations.
- Direct API providers: `claude-api`, `openai-api`, `bedrock`, `gemini-api`, `ollama-local`.
- Headless CLI providers: `claude-code`, `gemini-cli`, `codex-cli`, `antigravity-cli`, plus `claude-pty`.
- Hook surface for model-call, tool-call, step-boundary, and turn-done behavior.
- Per-turn timing instrumentation (`TurnTiming`) at the harness seam, broken
  down by round and tool, so the funnel can observe where a turn spends its wall clock.
- Shared subprocess plumbing for the CLI providers lives in `internal/subprocess`.
- `send_comms` is just a tool the funnel supplies — bridle has no special case.
- Funnel owns session JSONL; bridle proposes deltas.

## Provider categories

`direct-api` providers talk to a model API and let bridle own the tool loop.
They can consume bridle `ToolDef`s, run tools through the supplied
`ToolRunner`, fire before/after tool hooks, and use bridle MCP configuration.

`subprocess-stream` providers spawn a headless CLI that owns its own agent loop
and tool execution. Bridle observes the CLI JSON stream, emits model/tool
events, and returns a normalized `TurnResult`; it does not re-run those tool
calls through the bridle `ToolRunner`.

`codex-cli` uses `codex exec --json`. Set `TurnRequest.Model` to the Codex
model id to pass `--model`, or use `Model: "default"` to let the Codex CLI use
its configured default model while still satisfying bridle's required model
field. Existing sessions resume via `codex exec resume <session-id>`.

`antigravity-cli` drives the `agy` CLI headless (plain-text, not stream-json),
resuming via `-c` rather than a passed session id, and strips stale conversation
warnings from the output.

`ollama-local` exposes `KeepAlive` (how long the server keeps a model resident —
defaults to 30m so gemma-class models stay warm across quiet gaps), `NumCtx`
(the context window, mapped to `options.num_ctx`), and an `Options` map that is
merged into the request options on every turn.

## Test

```sh
go test ./...
```

Live provider smoke tests are opt-in. For Codex CLI:

```sh
BRIDLE_LIVE_CODEX=1 go test ./provider/codexcli -run TestRoundTripLive -count=1
```

## Non-goals

- Not a framework, not an agent runtime, not a session store. It does one thing: drive one turn.
- See spec §9 for the full non-goals list.

## Reference reading

PydanticAI (`iter()` / `CallToolsNode`), Eino (streaming + tool-call normalization), PI (headless JSON streaming), Strands (`register_hooks`), LangGraph (interrupt/checkpoint).
