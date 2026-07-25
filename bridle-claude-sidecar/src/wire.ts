// JSON-lines wire protocol between bridle (Go, provider/claudesdk) and
// this sidecar. Private to the pair — mirrors provider/claudesdk/wire.go
// field-for-field. Keep the two in sync when the wire changes, and bump
// PROTOCOL_VERSION (index.ts) so an older/newer peer rejects loudly at
// turn start instead of failing mid-turn (bridle-agentsdk-spec.md §5,§7).

export interface SidecarToolDef {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface SidecarInit {
  type: 'init';
  version: number;
  prompt: string;
  system_prompt_append?: string;
  model?: string;
  session_id?: string;
  resume?: string;
  max_turns?: number;
  cwd?: string;
  // Reasoning-effort ladder (agora-spec-bridle §3), passed through
  // verbatim to the Agent SDK's Options.effort — same vocabulary
  // ('low'|'medium'|'high'|'xhigh'|'max'), no translation needed.
  // Absent/undefined = provider default (option omitted).
  effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max';
  mode: 'funnel' | 'agent';
  allowed_tools?: string[];
  disallowed_tools?: string[];
  tools?: SidecarToolDef[];
  before_tool_call_gate?: boolean;
  extra_opts?: Record<string, unknown>;
}

export interface SidecarToolResult {
  type: 'tool_result';
  id: string;
  content: unknown;
  is_error: boolean;
}

export interface SidecarInterrupt {
  type: 'interrupt';
}

export type SidecarInbound = SidecarToolResult | SidecarInterrupt;

// --- outbound (sidecar -> bridle), one JSON object per stdout line ---

export interface EventTextDelta {
  type: 'text_delta';
  text: string;
}

export interface EventThinkingDelta {
  type: 'thinking_delta';
  text: string;
  signature?: string;
}

export interface EventToolCall {
  type: 'tool_call';
  id: string;
  name: string;
  args: unknown;
}

export interface EventNativeTool {
  type: 'native_tool';
  id: string;
  name: string;
  args: unknown;
  result?: unknown;
  is_error?: boolean;
}

export interface EventUsage {
  type: 'usage';
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

export interface EventDone {
  type: 'done';
  stop_reason: string;
  session_id?: string;
  model?: string;
}

export interface EventError {
  type: 'error';
  class: 'auth' | 'rate_limit' | 'provider' | 'refusal';
  message: string;
}

// claude.ai subscription plan-usage state — a WHOLLY SEPARATE SDK
// message type from EventError's "rate_limit" class above (that is an
// assistant turn failing; this is a usage reading that can fire on any
// turn, success or not, and never ends one). See bridle's RateLimit
// event doc comment (events.go) for the field semantics this mirrors.
export interface EventRateLimit {
  type: 'rate_limit_event';
  rate_limit_status: string;
  rate_limit_type: string;
  rate_limit_utilization: number;
  rate_limit_resets_at_ms: number;
  rate_limit_using_overage: boolean;
}

export type SidecarOutbound =
  | EventTextDelta
  | EventThinkingDelta
  | EventToolCall
  | EventNativeTool
  | EventUsage
  | EventDone
  | EventError
  | EventRateLimit;
