#!/usr/bin/env node
// bridle-claude-sidecar: a thin per-turn Node process wrapping
// @anthropic-ai/claude-agent-sdk, spoken to over stdio JSON-lines by
// bridle's Go provider (provider/claudesdk). Reads exactly one `init`
// line, drives one query(), and streams normalized events back on
// stdout until `done`/`error`. See wire.ts for the protocol and
// bridle-agentsdk-spec.md §5 for the design.
//
// v1 is per-turn spawn + resume (mirrors claudecode's session model) —
// this process handles exactly one turn and exits. No persistent daemon
// (spec §9 non-goal).

import { createInterface } from 'node:readline';
import { randomUUID } from 'node:crypto';
import {
  query,
  tool,
  createSdkMcpServer,
  type Options,
  type CanUseTool,
} from '@anthropic-ai/claude-agent-sdk';
import { jsonSchemaToZodShape } from './jsonSchema.js';
import type { SidecarInit, SidecarInbound, SidecarOutbound, EventError } from './wire.js';

// PROTOCOL_VERSION must match provider/claudesdk's protocolVersion
// constant (claudesdk.go). Bump both together on a breaking wire change.
const PROTOCOL_VERSION = 1;

function emit(event: SidecarOutbound): void {
  process.stdout.write(JSON.stringify(event) + '\n');
}

// currentQuery is the live query() handle for whatever turn is in
// flight, set inside main() below. Module-scope so the SIGTERM/SIGINT
// handler (registered once, below) can reach it without plumbing a
// reference through main()'s call chain.
let currentQuery: { interrupt(): Promise<unknown> } | undefined;

// Defense-in-depth SIGTERM/SIGINT handler (NEX-745 review gate, HIGH):
// bridle's Go side now kills this sidecar's whole PROCESS GROUP on
// cancellation (internal/subprocess.SetPgid + WatchCancelGroup), which
// is the primary fix and reaps the real `claude` CLI child even if this
// handler never runs. This handler is the second, independent layer: it
// gives the Agent SDK a chance to interrupt/dispose its own child
// cleanly (rather than relying solely on the OS-level group SIGKILL
// after the grace window) whenever this process receives the signal
// directly — e.g. a caller that sends SIGTERM without going through
// bridle's group-kill path, or a future wire-level `interrupt` message
// arriving concurrently with an external signal.
function installShutdownHandler(): void {
  let shuttingDown = false;
  const onSignal = (): void => {
    if (shuttingDown) return;
    shuttingDown = true;
    void (async () => {
      try {
        await currentQuery?.interrupt();
      } catch {
        // Best-effort: the query may already be done/gone. Exit either way.
      } finally {
        process.exit(0);
      }
    })();
  };
  process.on('SIGTERM', onSignal);
  process.on('SIGINT', onSignal);
}

function emitError(cls: EventError['class'], message: string): void {
  emit({ type: 'error', class: cls, message });
}

// classifySdkErrorMessage maps the Agent SDK's error-code vocabulary
// (SDKAssistantMessageError) onto bridle's wire error classes (spec §7).
function classifySdkErrorMessage(code: string | undefined): EventError['class'] {
  switch (code) {
    case 'authentication_failed':
    case 'oauth_org_not_allowed':
    case 'billing_error':
      return 'auth';
    case 'rate_limit':
    case 'overloaded':
      return 'rate_limit';
    default:
      return 'provider';
  }
}

interface PendingCustomCall {
  resolve: (v: { content: unknown; is_error: boolean }) => void;
}

interface PendingNativeCall {
  name: string;
  args: unknown;
}

async function main(): Promise<void> {
  installShutdownHandler();
  const rl = createInterface({ input: process.stdin, terminal: false });
  const lines = rl[Symbol.asyncIterator]();

  const first = await lines.next();
  if (first.done) {
    emitError('provider', 'stdin closed before an init line arrived');
    process.exitCode = 1;
    return;
  }

  let init: SidecarInit;
  try {
    init = JSON.parse(first.value as string) as SidecarInit;
  } catch (e) {
    emitError('provider', `malformed init line: ${(e as Error).message}`);
    process.exitCode = 1;
    return;
  }

  if (init.type !== 'init') {
    emitError('provider', `expected "init" as the first line, got type=${String((init as { type?: string }).type)}`);
    process.exitCode = 1;
    return;
  }
  // Version check FIRST, at turn start — a mismatch must never surface
  // mid-turn (spec §5,§7).
  if (init.version !== PROTOCOL_VERSION) {
    emitError(
      'provider',
      `protocol version mismatch: sidecar=${PROTOCOL_VERSION} caller=${init.version}`,
    );
    process.exitCode = 1;
    return;
  }

  // Pending custom-tool calls awaiting a tool_result from bridle, keyed
  // by the id THIS sidecar assigned when it emitted the tool_call.
  const pendingCustom = new Map<string, PendingCustomCall>();
  // Native (non-bridle-MCP) tool_use blocks awaiting their paired
  // tool_result in the next 'user' message, keyed by the SDK's own
  // tool_use id — observe-only (spec §5).
  const pendingNative = new Map<string, PendingNativeCall>();

  let liveQuery: { interrupt(): Promise<unknown> } | undefined;

  // Route every line after init to whichever inbound handler applies.
  // Runs concurrently with the query() consumption loop below.
  const routePromise = (async () => {
    for (;;) {
      const next = await lines.next();
      if (next.done) return;
      let msg: SidecarInbound;
      try {
        msg = JSON.parse(next.value as string) as SidecarInbound;
      } catch {
        continue; // malformed line — skip, don't crash the turn
      }
      if (msg.type === 'tool_result') {
        const pending = pendingCustom.get(msg.id);
        if (pending) {
          pendingCustom.delete(msg.id);
          pending.resolve({ content: msg.content, is_error: msg.is_error });
        }
      } else if (msg.type === 'interrupt') {
        void liveQuery?.interrupt();
      }
    }
  })();

  const toolDefs = (init.tools ?? []).map((t) =>
    tool(t.name, t.description, jsonSchemaToZodShape(t.input_schema), async (args: unknown) => {
      const id = randomUUID();
      const resultPromise = new Promise<{ content: unknown; is_error: boolean }>((resolve) => {
        pendingCustom.set(id, { resolve });
      });
      emit({ type: 'tool_call', id, name: t.name, args });
      const result = await resultPromise;
      const text = typeof result.content === 'string' ? result.content : JSON.stringify(result.content);
      return { content: [{ type: 'text' as const, text }], isError: result.is_error };
    }),
  );

  const mcpServers = toolDefs.length > 0 ? { bridle: createSdkMcpServer({ name: 'bridle', tools: toolDefs }) } : undefined;
  const mcpToolNames = (init.tools ?? []).map((t) => `mcp__bridle__${t.name}`);

  // Native tool exposure per mode (spec §3):
  //   funnel — ALL native tools off, Claude sees only bridle's tools.
  //   agent  — native toolset per allow/deny, bridle tools still reachable.
  let allowedTools: string[] | undefined;
  if (init.mode === 'funnel') {
    // Empty array (not undefined): an explicit allowlist of nothing, so
    // a funnel-mode turn with zero bridle tools defined is a plain
    // text-only turn with every native tool blocked — never falls back
    // to the CLI's default toolset.
    allowedTools = mcpToolNames;
  } else if (init.allowed_tools && init.allowed_tools.length > 0) {
    allowedTools = [...init.allowed_tools, ...mcpToolNames];
  } // else undefined: agent-mode default native toolset, mcp tools still reachable unrestricted

  // BeforeToolCall gating for CUSTOM (bridle mcp) tools happens bridle-side
  // (the harness's BeforeToolCallCtx hook fires inside executeToolCall) — those
  // tools are auto-approved via allowedTools and NORMALLY never reach
  // canUseTool at all. But the SDK's own Plan Mode (ExitPlanMode / the
  // client's plan-mode toggle) reroutes EVERY tool call — including
  // already-approved mcp ones — through canUseTool for the duration of
  // planning, as its own "nothing executes while planning" guarantee.
  // Without the mcpToolNames check below, that reroute hit the funnel-mode
  // deny-everything branch and blocked bridle's OWN gated tools
  // (read_file/write_file/run_command/question/...) mid-plan, using a
  // message written for genuinely-native tools — observed: every bridle
  // mcp call failing with "Native tool ... is disabled" while Plan Mode
  // was active, even though those tools aren't native at all.
  //
  // So in FUNNEL mode, a toolName NOT in mcpToolNames is always a NATIVE
  // tool (Read/Write/Edit/Bash/…), reaching canUseTool either directly or
  // via Plan Mode's reroute. Auto-allowing them (the old unconditional
  // `behavior: 'allow'`) let native tools run entirely OUTSIDE agora's
  // approval gate and containment — defeating funnel mode's own contract
  // ("all native tools off; Claude sees only bridle's tools"). DENY them:
  // the model is pushed onto bridle's gated, sandboxed tools. A toolName
  // that IS in mcpToolNames is one of bridle's own — allow it through to
  // bridle's real (BeforeToolCall) gate rather than dying here, whether it
  // arrived via the normal bypass or Plan Mode's reroute. In agent mode we
  // keep the default-allow (native toolset is intentional there).
  const canUseTool: CanUseTool = async (toolName, input) => {
    if (init.mode === 'funnel' && !mcpToolNames.includes(toolName)) {
      return {
        behavior: 'deny',
        message: `Native tool "${toolName}" is disabled in this session — use the bridle tools (read_file, write_file, run_command).`,
      };
    }
    return { behavior: 'allow', updatedInput: input };
  };

  // Hard block: allowedTools/permissionMode/canUseTool do NOT keep the SDK's
  // native toolset out of a funnel-mode turn (verified: native Bash still ran).
  // disallowedTools is the only reliable lever — list every built-in so the
  // model is forced onto bridle's gated mcp tools.
  const NATIVE_TOOLS = ['Task', 'Bash', 'BashOutput', 'KillShell', 'KillBash', 'Glob', 'Grep', 'Read', 'Edit', 'MultiEdit', 'Write', 'NotebookEdit', 'WebFetch', 'WebSearch', 'TodoWrite', 'ExitPlanMode', 'SlashCommand', 'ListMcpResources', 'ReadMcpResource'];
  const effectiveDisallowed = init.mode === 'funnel'
    ? [...NATIVE_TOOLS, ...(init.disallowed_tools ?? [])]
    : init.disallowed_tools;

  const options: Options = {
    systemPrompt: init.system_prompt_append
      ? { type: 'preset', preset: 'claude_code', append: init.system_prompt_append }
      : { type: 'preset', preset: 'claude_code' },
    model: init.model,
    maxTurns: init.max_turns,
    cwd: init.cwd,
    allowedTools,
    disallowedTools: effectiveDisallowed,
    // Funnel mode = agora owns ALL tools + approvals. 'dontAsk' means "deny any
    // tool that isn't pre-approved" — and only the bridle mcp tools are
    // pre-approved (allowedTools). So every NATIVE tool (Read/Write/Edit/Bash/…)
    // is denied WITHOUT running, forcing the model onto bridle's gated,
    // sandboxed tools. The default mode does NOT do this: it silently ran native
    // Bash/Write outside agora's approval gate (canUseTool was never even
    // invoked for them). Agent mode keeps the CLI's normal permission flow.
    permissionMode: init.mode === 'funnel' ? 'dontAsk' : undefined,
    mcpServers,
    canUseTool,
    resume: init.resume,
    sessionId: init.session_id,
  };
  // Reasoning-effort ladder (agora-spec-bridle §3): the SDK's Options.effort
  // uses the SAME vocabulary bridle's wire carries ('low'|'medium'|'high'|
  // 'xhigh'|'max'), so this is a straight passthrough, no translation.
  // Only set when non-empty — an explicit `effort: undefined` still shows up
  // as an own-property on the options object, which is harmless here but
  // the guard keeps the omitted-means-default contract obvious.
  if (init.effort) {
    options.effort = init.effort;
  }

  if (init.extra_opts) {
    Object.assign(options, init.extra_opts);
  }

  let sessionId: string | undefined;
  let resolvedModel: string | undefined;
  let sawResult = false;

  try {
    const q = query({ prompt: init.prompt, options });
    liveQuery = q;
    currentQuery = q;

    for await (const msg of q) {
      const anyMsg = msg as Record<string, unknown>;

      if (anyMsg.type === 'assistant') {
        const message = anyMsg.message as { content?: unknown[]; model?: string } | undefined;
        const errCode = anyMsg.error as string | undefined;
        if (errCode) {
          emitError(classifySdkErrorMessage(errCode), `assistant error: ${errCode}`);
          continue;
        }
        if (message?.model) resolvedModel = message.model;
        for (const rawBlock of message?.content ?? []) {
          const block = rawBlock as Record<string, unknown>;
          switch (block.type) {
            case 'text':
              if (typeof block.text === 'string' && block.text) {
                emit({ type: 'text_delta', text: block.text });
              }
              break;
            case 'thinking':
              if (typeof block.thinking === 'string' && block.thinking) {
                emit({
                  type: 'thinking_delta',
                  text: block.thinking,
                  signature: typeof block.signature === 'string' ? block.signature : undefined,
                });
              }
              break;
            case 'tool_use': {
              const name = block.name as string;
              const id = block.id as string;
              // Bridle's own MCP tools are already fully handled by the
              // tool() handler closure above (it emits its own
              // "tool_call" and awaits "tool_result" directly) — only
              // OTHER (native / non-bridle-MCP) tool_use blocks are
              // observed here, matching spec §5's native_tool
              // (observe-only, drives AfterToolCall) event.
              if (typeof name === 'string' && !name.startsWith('mcp__bridle__')) {
                pendingNative.set(id, { name, args: block.input });
              }
              break;
            }
            default:
              break;
          }
        }
      } else if (anyMsg.type === 'user') {
        const message = anyMsg.message as { content?: unknown[] } | undefined;
        for (const rawBlock of message?.content ?? []) {
          const block = rawBlock as Record<string, unknown>;
          if (block.type === 'tool_result') {
            const toolUseId = block.tool_use_id as string;
            const pendingCall = pendingNative.get(toolUseId);
            if (pendingCall) {
              pendingNative.delete(toolUseId);
              emit({
                type: 'native_tool',
                id: toolUseId,
                name: pendingCall.name,
                args: pendingCall.args,
                result: block.content,
                is_error: Boolean(block.is_error),
              });
            }
          }
        }
      } else if (anyMsg.type === 'result') {
        sawResult = true;
        const usage = anyMsg.usage as
          | {
              input_tokens?: number;
              output_tokens?: number;
              cache_read_input_tokens?: number;
              cache_creation_input_tokens?: number;
            }
          | undefined;
        emit({
          type: 'usage',
          input_tokens: usage?.input_tokens ?? 0,
          output_tokens: usage?.output_tokens ?? 0,
          cache_read_input_tokens: usage?.cache_read_input_tokens ?? 0,
          cache_creation_input_tokens: usage?.cache_creation_input_tokens ?? 0,
        });

        sessionId = (anyMsg.session_id as string) ?? sessionId;
        const subtype = anyMsg.subtype as string | undefined;
        const stopReason = (anyMsg.stop_reason as string | null) ?? (subtype === 'success' ? 'end_turn' : subtype ?? 'error');

        if (subtype && subtype !== 'success') {
          // SDKResultError: surface as a provider-class error rather
          // than a clean "done" — spec §7 maps "SDK/CLI crash or
          // protocol mismatch" to the provider class; a non-success
          // result subtype (error_during_execution, error_max_turns,
          // etc.) is the SDK's own terminal-failure signal.
          const errors = (anyMsg.errors as string[] | undefined) ?? [];
          emitError('provider', `sdk result subtype=${subtype}: ${errors.join('; ')}`);
        } else {
          emit({
            type: 'done',
            stop_reason: stopReason ?? 'end_turn',
            session_id: sessionId,
            model: resolvedModel,
          });
        }
      }
    }

    if (!sawResult) {
      emitError('provider', 'query() stream ended without a result message');
      process.exitCode = 1;
    }
  } catch (e) {
    const err = e as Error;
    emitError('provider', `query() threw: ${err.message}`);
    process.exitCode = 1;
  } finally {
    rl.close();
    void routePromise;
  }
}

main().catch((e: unknown) => {
  emitError('provider', `sidecar crashed: ${(e as Error).message}`);
  process.exitCode = 1;
});
