package claudesdk

import "encoding/json"

// This file defines the JSON-lines wire protocol between bridle
// (provider/claudesdk) and the bridle-claude-sidecar Node process. The
// protocol is private to this package (bridle-agentsdk-spec.md §5) — no
// other bridle package or caller should depend on these shapes. The
// TypeScript sidecar (bridle-claude-sidecar/src/index.ts) mirrors these
// field names exactly; keep the two in sync when the wire changes, and
// bump protocolVersion (claudesdk.go) so a mismatched sidecar rejects
// loudly at turn start instead of failing mid-turn (spec §5,§7).

// sidecarToolDef is one bridle ToolDef lowered onto the wire, passed
// through to the sidecar's SDK-MCP tool() registration VERBATIM — name,
// description, and JSON-schema must not be remangled (spec §5).
type sidecarToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// sidecarInit is the one line bridle sends to open a turn. Sent exactly
// once, before anything else reaches the sidecar's stdin.
type sidecarInit struct {
	Type string `json:"type"` // "init"

	// Version guards wire compatibility (spec §5,§7): the sidecar
	// rejects a mismatch loudly at turn start (an "error" event with
	// class "provider"), never mid-turn.
	Version int `json:"version"`

	Prompt             string `json:"prompt"`
	SystemPromptAppend string `json:"system_prompt_append,omitempty"`
	Model              string `json:"model,omitempty"`

	// Resume/SessionID mirror claudecode's --resume/--session-id split
	// (claudecode.go's package doc): SessionID is set when the funnel is
	// starting a FRESH session under a caller-chosen id (SDK's
	// `sessionId` option); Resume is set when continuing an existing one
	// (SDK's `resume` option). At most one is non-empty.
	SessionID string `json:"session_id,omitempty"`
	Resume    string `json:"resume,omitempty"`

	MaxTurns int    `json:"max_turns,omitempty"`
	Cwd      string `json:"cwd,omitempty"`

	// Effort mirrors ProviderRequest.Effort, passed through verbatim to
	// the SDK's Options.effort (sdk.d.ts: EffortLevel = 'low' | 'medium'
	// | 'high' | 'xhigh' | 'max' — the SAME ladder agora-spec-bridle §3
	// uses, no translation needed on this lane). Empty = provider
	// default (index.ts omits the option entirely).
	Effort string `json:"effort,omitempty"`

	// Mode is "funnel" (all native tools off, Claude sees only Tools) or
	// "agent" (native toolset per AllowedTools/DisallowedTools) — spec §3.
	Mode            string           `json:"mode"`
	AllowedTools    []string         `json:"allowed_tools,omitempty"`
	DisallowedTools []string         `json:"disallowed_tools,omitempty"`
	Tools           []sidecarToolDef `json:"tools,omitempty"`

	// BeforeToolCallGate is currently a DEAD/reserved field (NEX-745
	// review gate, MED — "dead gate" finding): v1's index.ts canUseTool
	// does NOT read it — canUseTool unconditionally allows every native
	// tool call regardless of this value. Bridle's real BeforeToolCall
	// enforcement for CUSTOM tools happens entirely on the Go side, via
	// the tool_call/tool_result round trip serviced by
	// req.ToolExecutor -> Harness.executeToolCall — this field plays no
	// part in that path either. It is sent (always true) as a forward-
	// compatible placeholder for a FUTURE canUseTool->bridle round trip
	// that would let a BeforeToolCall hook veto a native (ModeAgent)
	// tool call; that round trip does not exist yet. See
	// claudesdk.go's Capabilities() (SupportsBeforeToolCall is false in
	// ModeAgent precisely because this field doesn't do anything there)
	// and bridle-claude-sidecar/README.md's "canUseTool always
	// default-allows" note.
	BeforeToolCallGate bool `json:"before_tool_call_gate"`

	// ExtraOpts is passed through to `options` verbatim, for opts this
	// wire doesn't have a named field for yet (Config.ExtraOpts).
	ExtraOpts map[string]json.RawMessage `json:"extra_opts,omitempty"`
}

// sidecarToolResult is bridle's reply to a "tool_call" event: the
// executed result, keyed by the call id the sidecar assigned.
type sidecarToolResult struct {
	Type    string          `json:"type"` // "tool_result"
	ID      string          `json:"id"`
	Content json.RawMessage `json:"content"`
	IsError bool            `json:"is_error"`
}

// sidecarInterrupt is the in-band interrupt message (spec §5). bridle's
// Go provider does not send this in v1 — cancellation goes through
// internal/subprocess.WatchCancel (SIGTERM, grace, SIGKILL), exactly the
// claudecode path — but the message is part of the documented wire and
// the sidecar honours it if a future caller sends it directly.
type sidecarInterrupt struct {
	Type string `json:"type"` // "interrupt"
}

// sidecarEvent is the union of everything the sidecar sends back on
// stdout, one JSON object per line. Type discriminates the variant;
// only the fields relevant to that Type are populated.
type sidecarEvent struct {
	Type string `json:"type"`

	// text_delta / thinking_delta
	Text      string `json:"text,omitempty"`
	Signature string `json:"signature,omitempty"` // thinking_delta only

	// tool_call (custom, bridle must execute) / native_tool (observe-only)
	ID      string          `json:"id,omitempty"`
	Name    string          `json:"name,omitempty"`
	Args    json.RawMessage `json:"args,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"` // native_tool only
	IsError bool            `json:"is_error,omitempty"`

	// usage
	InputTokens              int `json:"input_tokens,omitempty"`
	OutputTokens             int `json:"output_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`

	// done
	StopReason string `json:"stop_reason,omitempty"`
	SessionID  string `json:"session_id,omitempty"` // echoed resume id (spec §7 invariant 3)
	Model      string `json:"model,omitempty"`      // upstream-resolved model id

	// error
	Class   string `json:"class,omitempty"` // auth | rate_limit | provider | refusal
	Message string `json:"message,omitempty"`
}
