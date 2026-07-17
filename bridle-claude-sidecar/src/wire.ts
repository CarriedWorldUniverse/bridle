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

export type SidecarOutbound =
  | EventTextDelta
  | EventThinkingDelta
  | EventToolCall
  | EventNativeTool
  | EventUsage
  | EventDone
  | EventError;
