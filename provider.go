package bridle

import "context"

// ProviderCategory classifies how a provider executes tool calls.
type ProviderCategory string

const (
	// CategoryDirectAPI — provider talks directly to a model API; bridle owns the tool loop.
	CategoryDirectAPI ProviderCategory = "direct-api"
	// CategorySubprocessStream — provider spawns a subprocess that runs its own agentic loop
	// and emits a structured event stream. The subprocess owns tool execution.
	CategorySubprocessStream ProviderCategory = "subprocess-stream"
)

// ProviderCapabilities advertises what a provider supports so the harness
// and funnel can route turns correctly.
type ProviderCapabilities struct {
	Category               ProviderCategory
	SupportsCustomTools    bool // funnel can pass arbitrary Tools via TurnRequest
	SupportsBeforeToolCall bool // BeforeToolCall hook fires
	SupportsAfterToolCall  bool // AfterToolCall hook fires
	SupportsMCP            bool // provider consumes TurnRequest.MCP (direct-api only)
}

// Provider is the interface every model backend must implement.
// Provider-specific weirdness (streaming, wire format, tool-schema translation)
// stays inside the implementation; the harness sees a uniform event stream.
type Provider interface {
	Name() ProviderID
	Capabilities() ProviderCapabilities
	RunTurn(ctx context.Context, req ProviderRequest, sink EventSink) (ProviderResult, error)
}

// ProviderRequest is the harness-internal lowered form of TurnRequest.
// System prompt is assembled, session tail is flattened to the provider's
// message format, tools are translated to provider-specific schema, and
// inbox items are folded in.
type ProviderRequest struct {
	AspectID     string
	AppendSystemPrompt string
	Session      SessionHandle  // for subprocess-stream: resume key; for direct-api: may be empty
	Messages     []ProviderMessage
	Tools        []ToolDef
	ToolChoice   string            // see TurnRequest.ToolChoice
	MCP          *MCPClientConfig  // nil = no MCP tools
	MaxSteps     int
	Model        string

	// Cwd is the working directory for subprocess-style providers (see
	// TurnRequest.Cwd). Empty falls through to bridle's host cwd. Direct-
	// API providers ignore this field.
	Cwd string

	// ProviderEnv is per-turn auth/routing env (see TurnRequest.ProviderEnv).
	// Subprocess providers overlay it onto the spawned process's env;
	// direct-API providers read it as auth/base-url config. Empty/nil =
	// provider uses its own default config.
	ProviderEnv map[string]string
}

// ProviderMessage is a single exchange entry in provider-agnostic form.
//
// For Role == "tool_result", both ToolCallID and ToolName must be set.
// ToolCallID is the call instance identifier the assistant emitted (used
// to correlate this result with that specific invocation). ToolName is
// the function-declaration name that was called (e.g. "send_chat") —
// some providers (Gemini's FunctionResponse) require it to be present
// alongside the call id, because their wire format keys responses by
// declaration name, not by call id. Providers that key only by call id
// (Anthropic, OpenAI, Ollama) ignore ToolName and the field can be left
// empty without harm.
//
// For Role == "assistant", ToolCalls carries the structured tool_use
// blocks the model emitted on this turn. Providers that send assistant
// history back to the model (claude, openai, gemini, bedrock) MUST
// reconstruct these as native tool_use blocks; sending only Content as
// plain text loses the tool-call structure and breaks multi-turn tool
// conversations on strict providers (Bedrock rejects, Anthropic and
// OpenAI are lenient but degrade). Content and ToolCalls can both be
// non-empty — text and tool_use blocks coexist in one assistant turn.
type ProviderMessage struct {
	Role       string // "user" | "assistant" | "tool_result" | "system"
	Content    string
	ToolCallID string           // links a tool_result back to the call that produced it
	ToolName   string           // function-declaration name; required for tool_result on Gemini
	ToolCalls  []ToolInvocation // tool_use blocks for assistant turns; nil on other roles
}

// ProviderResult is the harness-internal result from one provider turn step.
//
// FinalText is the model's settled assistant text for this round —
// what downstream consumers (e.g. nexus funnel auto-post to chat)
// should treat as "what the model said." The harness concatenates
// FinalText across rounds for direct-API providers (see run.go),
// because each round is a separate, intentional deliberation.
//
// Subprocess-stream providers that parse multi-event streams must
// decide what counts as "settled" before populating FinalText. Some
// models (Claude trained for claudecode) produce a draft → tool → final
// answer pattern within one subprocess run; the draft is exploratory
// and should NOT survive into FinalText, or auto-post emits a doubled
// "draft + rewrite" row (operator chat #951, harrow #944). claudecode
// handles this by resetting accumulated text on every tool_use, so
// FinalText ends up containing only post-last-tool text.
//
// Other subprocess-stream providers (geminicli) don't currently apply
// the same heuristic — that's an explicit per-model judgement, not a
// missed fix. Revisit if a model exhibits the draft-rewrite pattern
// without this policy.
type ProviderResult struct {
	FinalText    string
	ToolCalls    []ToolInvocation
	StepCount    int
	Usage        Usage
	StopReason   StopReason
	SessionDelta []SessionEvent
	// ResolvedModel is the model id the upstream API actually returned
	// (e.g. "claude-3-5-sonnet-20241022"). May differ from
	// ProviderRequest.Model when per-turn ProviderEnv routed the call
	// elsewhere. Empty when the provider doesn't surface a model id.
	// Flows into TurnResult.ResolvedModel.
	ResolvedModel string
}
