package bridle

import (
	"context"
	"encoding/json"
	"errors"
	"time"
)

// ErrModelRequired is returned by RunTurn when TurnRequest.Model is empty.
var ErrModelRequired = errors.New("bridle: TurnRequest.Model is required")

// ProviderID identifies a model provider.
type ProviderID string

const (
	ProviderClaude         ProviderID = "claude-api"
	ProviderClaudeCode     ProviderID = "claude-code"
	ProviderClaudeSDK      ProviderID = "claude-sdk"
	ProviderClaudePty      ProviderID = "claude-pty"
	ProviderOllama         ProviderID = "ollama-local"
	ProviderOpenAI         ProviderID = "openai-api"
	ProviderBedrock        ProviderID = "bedrock"
	ProviderGemini         ProviderID = "gemini-api"
	ProviderGeminiCLI      ProviderID = "gemini-cli"
	ProviderCodexCLI       ProviderID = "codex-cli"
	ProviderAntigravityCLI ProviderID = "antigravity-cli"
)

// StopReason explains why a turn ended.
type StopReason string

const (
	StopReasonModelDone StopReason = "model_done"
	StopReasonMaxSteps  StopReason = "max_steps"
	StopReasonError     StopReason = "error"
	StopReasonAborted   StopReason = "aborted"
	// StopReasonProcessExit is set when the underlying provider process
	// exited non-zero AFTER producing parseable assistant content. The
	// returned ProviderResult carries whatever the model said before
	// the exit — callers should treat the result as truncated-but-real,
	// not discard it. Common cause: hitting an output-token cap and the
	// CLI surfacing that as a non-zero exit rather than a clean stop.
	StopReasonProcessExit StopReason = "process_exit"

	// StopReasonRefusal is a vocabulary value for a model-declined-to-
	// answer stop (agora-spec-bridle §2 done{stop_reason:refusal};
	// Anthropic's Messages API surfaces this as its own stop_reason on
	// Fable 5, HTTP 200). NEX-767 T1/T7 adds the CONST so Stream's
	// done{refusal} mapping (see stream.go's streamStopReason) exists
	// and is testable; no provider's wire-to-StopReason mapping
	// produces it yet — detecting the real wire signal per lane is T3
	// follow-up work (agora-spec-bridle §3's refusal-handling item).
	StopReasonRefusal StopReason = "refusal"
)

// Usage holds token and cost data for a turn.
//
// InputTokens is the count of UNCACHED prompt tokens billed at full
// rate. CacheReadInputTokens and CacheCreationInputTokens surface
// claude-api's prompt-caching behavior — the former is read at a
// discount, the latter is the new content being added to cache. Cache
// fields are zero for providers that don't expose caching (or don't
// run a cache-eligible request).
//
// Sum (InputTokens + CacheReadInputTokens + CacheCreationInputTokens)
// approximates the total prompt size the model received. Use that
// for context-fullness reasoning; use InputTokens alone for billing
// estimates of fresh content.
type Usage struct {
	InputTokens              int
	OutputTokens             int
	CacheReadInputTokens     int     // Anthropic prompt-cache hit count
	CacheCreationInputTokens int     // tokens written into the prompt cache this turn
	CostUSD                  float64 // provider-reported or estimated; 0 if unknown

	// Estimated is set true when the token counts in this Usage were
	// NOT reported by the engine — they were estimated by bridle's
	// tokenizer as a last-resort floor (the usage contract, NEX-581).
	// A provider that reports real usage leaves this false. The flag
	// rides through addUsage: if ANY round of a turn was estimated, the
	// turn total is flagged Estimated. Consumers (cost accounting) can
	// treat estimated counts as approximate. The guarantee is that a
	// completed turn never has silently-zero usage — it has real
	// counts, or a flagged estimate, never nothing.
	Estimated bool

	// ReasoningTokens is the count of extended-thinking/reasoning
	// tokens the provider billed for this round, when it reports the
	// breakdown separately from OutputTokens (agora-spec-bridle §2
	// usage{input, output, cached, reasoning}). Additive field; 0 for
	// providers/rounds that don't report it (not necessarily "no
	// reasoning happened" — it may just be folded into OutputTokens
	// instead, e.g. Anthropic bills thinking tokens as output tokens
	// today).
	ReasoningTokens int
}

// ToolInvocation records a single tool call the model made.
type ToolInvocation struct {
	ID     string
	Name   string
	Args   json.RawMessage
	Result json.RawMessage
	Err    string
}

// InboxItem is a comms message that arrived during the previous turn.
// The harness folds these into the prompt context before the first model call.
// Read-only from the harness's perspective.
//
// MsgID is the chat msg_id this item was sourced from. It carries
// through into the prompt so the model can reference items by id when
// triaging ("triage(msg_id=17, decision='reply')"). Zero means the
// item didn't originate from a chat message — it's an internal/synthetic
// item the funnel injected, and the triage contract doesn't apply.
type InboxItem struct {
	From    string
	Content string
	MsgID   int64
	RawJSON json.RawMessage

	// ThreadRoot is the canonical thread identity for the message
	// (linked-list root id; nexus task #226). The funnel uses it to
	// key per-thread session state so each thread gets its own
	// claude-code jsonl, preventing SessionTail bleed across threads.
	// Zero = legacy/non-chat synthetic item or pre-#226 row.
	ThreadRoot int64

	// Source identifies which trigger channel produced this item.
	// Empty defaults to "chat" (legacy / nexus-chat substrate).
	// agora-side callers set Source="tty" for operator-typed inputs,
	// allowing ReturnHandlers to branch routing (chat → bus reply,
	// tty → panel-only). Future trigger channels add new Source
	// values; consumers default-treat unknown values as "chat" for
	// backward compat.
	Source string
}

// TurnRequest is the complete input for one deliberation turn.
type TurnRequest struct {
	// Identity & framing
	AspectID           string           // who's running (cost/triage/identity attribution)
	AppendSystemPrompt string           // composed by funnel: NEXUS.md + SOUL.md + PRIMER + harness rules
	SystemPromptMode   SystemPromptMode // how AppendSystemPrompt is applied: append (default, zero value) extends claude-code's base prompt; replace swaps it entirely
	Session            SessionHandle    // opaque handle for provider-side state (subprocess-stream: resume key)
	SessionTail        []SessionEvent   // recent events for direct-api providers to lower into the request

	// This turn
	UserMessage string      // the prompt that opens this turn (may be empty for autonomous)
	Inbox       []InboxItem // mid-turn comms accumulated since last turn

	// Tool surface
	Tools []ToolDef        // explicit in-process tool defs
	MCP   *MCPClientConfig // MCP-loaded tools; nil = no MCP tools this turn

	// Provider
	Provider ProviderID // claude-api | openai-api | bedrock | gemini-api | ollama-local | claude-code | gemini-cli | codex-cli | claude-pty
	Model    string     // REQUIRED — provider-specific model id; RunTurn returns ErrModelRequired if empty
	MaxSteps int        // hard cap on tool-call rounds; 0 = unlimited

	// ToolChoice optionally constrains how the model picks tools.
	// Empty string → provider default (typically "auto").
	// "auto" → model decides whether to call a tool.
	// "any" → model must call exactly one tool, free choice of which.
	// "none" → no tools may be called this turn (text only).
	// Any other value → name of a specific tool that must be called.
	// Not all providers honour all values; unsupported values fall back to "auto".
	ToolChoice string

	// Sampling controls (NEX-299 Pass 2). Pointer types so "unset" is
	// distinguishable from "explicitly zero" — providers fall through
	// to their own default when nil.
	//
	//   Temperature: lower = more deterministic. 0 for classifier
	//                tasks (cheap judge); higher for creative.
	//   TopP:        nucleus sampling. Standard across providers.
	//   TopK:        claude-only; openai silently ignores.
	//   Seed:        openai-only deterministic sampling seed;
	//                claude silently ignores. Pair with Temperature=0
	//                for full reproducibility.
	Temperature *float64
	TopP        *float64
	TopK        *int
	Seed        *int

	// ThinkingBudgetTokens requests Anthropic extended-thinking with this token
	// budget. Anthropic requires it be >= 1024 and < the request's max_tokens.
	// 0 = unset/disabled (provider default, no thinking). Claude-only; other
	// providers ignore it, same as Seed/TopK precedent.
	ThinkingBudgetTokens int

	// Effort is the agora reasoning-effort ladder value: low | medium |
	// high | xhigh | max (agora-spec-bridle §3). Empty = provider
	// default (no translation attempted). Providers that can't express
	// a tier at all drop it silently, same as Seed/TopK precedent.
	Effort string

	// MaxOutputTokens caps generation length. 0 = provider default
	// (claude internally falls back to 4096; openai uses its own
	// account-level default). Set non-zero for cost-bounded paths
	// like cheap-judge classifier where verdicts are tiny.
	MaxOutputTokens int

	// StopSequences halt generation on first match. Maps to openai
	// `stop` and claude `stop_sequences`. Empty = no stop sequences.
	StopSequences []string

	// ResponseFormat constrains the model's output shape — most
	// usefully for json_schema strict mode, which guarantees the
	// response matches Schema. Providers that don't support this
	// (claude as of writing) silently ignore. Callers wanting
	// portability should also encode schema requirements in the
	// system prompt. Nil = free-form text (provider default).
	ResponseFormat *ResponseFormat

	// ToolCallStrictness is the per-aspect tool-call contract knob
	// (NEX-581). It controls how hard bridle works to deliver a clean
	// tool call or clean text when an engine leaks raw protocol tokens.
	// Empty = repair-then-retry (the default): builders should keep it
	// strict and never ship a degraded tool call; research aspects can
	// set "tolerant" to accept structurally-repaired text without a
	// retry round.
	ToolCallStrictness ToolCallStrictness

	// Cwd is the working directory for subprocess-style providers.
	// Empty falls through to the bridle host process's cwd. Per-request
	// rather than per-Harness because different aspects sharing one
	// Harness need distinct cwds. For example, claude-code derives its
	// session jsonl path AND its .mcp.json discovery from cwd, so two
	// aspects with the same Harness but overlapping cwds collide
	// sessions and leak MCP identity from one into the other. Codex CLI
	// receives the same value as both process cwd and `codex --cd`.
	// Direct-API providers ignore this field — they have no subprocess
	// to anchor.
	Cwd string

	// ProviderEnv is per-call environment for the provider. Direct-API
	// providers read it as their auth/routing config (commonly
	// ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, OPENAI_API_KEY,
	// OPENAI_BASE_URL); subprocess providers propagate it into the
	// spawned process's env so the same per-turn override pattern
	// applies. nil/empty = use whatever the provider already has on its
	// own (process env, --bare-style flags, etc).
	//
	// Per-call rather than per-process so a single funnel can mix
	// credentials across turns — e.g. main turn against the operator's
	// Anthropic credit pool, judge turn against a DeepSeek-via-
	// Anthropic-shape credential, eval turn against OpenAI. The
	// credential store wires this from aspects.default_*_credential
	// per task #218.
	ProviderEnv map[string]string

	// ContextPolicy is the per-aspect context-window policy (the context
	// contract, NEX-581): a desired window (TargetWindow) and a soft
	// prompt-size budget (PromptBudget), expressed once here and mapped
	// by each provider to its engine knob — or warned on engine-
	// agnostically. Zero value = no policy (engine defaults, no budget
	// warning). See ContextPolicy.
	ContextPolicy ContextPolicy
}

// TurnResult is the structured outcome of a completed turn.
//
// ResolvedModel is the model identifier the upstream API actually
// returned (Anthropic Messages.Model, OpenAI ChatCompletion.Model,
// claudecode result-event model, etc.). It can differ from
// TurnRequest.Model when per-turn ProviderEnv routes the call to a
// different backend — operator pool's Claude credit vs. DeepSeek-via-
// Anthropic-shape credential vs. OpenAI. Empty when the provider
// doesn't surface a model id. Callers attributing usage/cost/identity
// should prefer ResolvedModel when non-empty, fallback to
// TurnRequest.Model.
type TurnResult struct {
	FinalText     string           // model's last assistant text (may be empty for tool-only turns)
	ToolCalls     []ToolInvocation // ordered list of what the model actually did
	StepCount     int
	Usage         Usage
	StopReason    StopReason
	ResolvedModel string         // model id the upstream API reported; empty when unknown
	SessionDelta  []SessionEvent // events to propose to the funnel-owned JSONL
	Timing        TurnTiming     // per-turn timing instrumentation; zero value = not recorded
}

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

// EventSink receives events as the turn unfolds.
type EventSink interface {
	Emit(Event)
}

// Harness drives one deliberation turn with one provider.
type Harness struct {
	provider Provider
	hooks    hookRegistry
	now      func() time.Time // injectable clock; nil means time.Now
}

// clock returns the harness's time source: the injected now func when
// set (tests), time.Now otherwise.
func (h *Harness) clock() func() time.Time {
	if h.now != nil {
		return h.now
	}
	return time.Now
}

// NewHarness creates a Harness backed by the given provider.
func NewHarness(p Provider) *Harness {
	return &Harness{provider: p}
}

// RunTurn drives one turn: calls the provider, executes tool calls via runner,
// fires hooks at documented points, and emits events to sink.
// Cancellation via ctx returns a partial TurnResult with StopReason=aborted.
// Timing is populated on normal completion and on provider errors; it is
// zero on context-cancellation aborts.
// Returns ErrModelRequired if req.Model is empty.
func (h *Harness) RunTurn(ctx context.Context, req TurnRequest, runner ToolRunner, sink EventSink) (result TurnResult, err error) {
	if req.Model == "" {
		return TurnResult{StopReason: StopReasonError}, ErrModelRequired
	}
	defer func() {
		if r := recover(); r != nil {
			e := panicErr(r)
			// Stamp directly: this emit bypasses runTurn's stampSink
			// (the panic unwound past it), so TS would otherwise be zero.
			sink.Emit(TurnError{Err: e, Stage: TurnErrorStageHarnessRecover, TS: h.clock()()})
			result.StopReason = StopReasonError
			err = e
		}
	}()
	return h.runTurn(ctx, req, runner, sink)
}
