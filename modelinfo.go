package bridle

// LaneID identifies a fine-grained model lane — finer than ProviderID.
//
// ProviderID isn't fine-grained enough for the registry: openai.go
// returns ProviderOpenAI regardless of baseURL, so OpenAI-proper and
// DeepSeek collide under one ProviderID key (DeepSeek is the same Go
// Provider pointed at api.deepseek.com). The registry keys catalog rows
// and Harness bindings on LaneID instead, one level below ProviderID.
type LaneID string

// The 7 core lanes the T1 static catalog covers. Uncataloged lanes are
// a deliberate gap, not an oversight — Registry.Resolve errors "not
// cataloged" for anything outside this set, which enforces
// core-lanes-only without needing a separate allowlist.
const (
	LaneClaudeAPI  LaneID = "claude-api"
	LaneClaudeCode LaneID = "claude-code"
	LaneClaudeSDK  LaneID = "claude-sdk" // NEX-745 (parallel branch) owns the Provider; catalog rows here are placeholders
	LaneOpenAI     LaneID = "openai"
	LaneDeepSeek   LaneID = "deepseek"
	LaneGeminiAPI  LaneID = "gemini-api"
	LaneBedrock    LaneID = "bedrock"
	LaneOllama     LaneID = "ollama"
)

// ModelCapabilities are model-level capability axes — orthogonal to the
// per-provider ProviderCapabilities (which describe how a Go Provider
// executes tool calls: subprocess-stream vs direct-api, MCP support,
// etc). A model's EFFECTIVE tool support is the AND of both:
// ProviderCapabilities.SupportsCustomTools && ModelCapabilities.Tools.
// The remaining fields here have no ProviderCapabilities analogue —
// they're net-new axes agora needs (agora-spec-bridle §1).
type ModelCapabilities struct {
	Tools            bool
	ParallelTools    bool
	Streaming        bool
	ReasoningEffort  bool
	StructuredOutput bool
	PromptCaching    bool
	Vision           bool

	// SystemPromptMode advertises which system-prompt application mode
	// this catalog row was authored against: SystemPromptFull ("full",
	// alias of Replace) or SystemPromptAppend ("append"). claude-code
	// genuinely supports BOTH — the catalog row's "append" is an
	// ADVISORY default recording agora policy (preserve claude-code's
	// built-in prompt), not a hard capability gate. Callers may still
	// pass either mode on TurnRequest/ProviderRequest; this field only
	// documents the lane's typical/recommended choice.
	SystemPromptMode SystemPromptMode
}

// ModelPricing is per-million-token USD pricing. Zero value means
// pricing is unknown/unset — token-only accounting until populated
// (agora-spec-bridle §1: "optional pricing {in, out, cached} — enables
// cost-aware workflow budgets — token-only until present").
type ModelPricing struct {
	In     float64
	Out    float64
	Cached float64
}

// PromptDialect carries model-global presentation knobs — tool idiom,
// wire format, thinking guidance. Per-core adjustments/renditions live
// in the agora core package, not here (agora-spec-prompt §2a/§4).
type PromptDialect struct {
	ToolIdiom string
	Format    string
	Thinking  string
}

// ModelPromptMeta is the optional prompt-presentation metadata for a
// catalog row (agora-spec-bridle §1: "optional prompt {dialect |
// rendition_ref}").
type ModelPromptMeta struct {
	Dialect      *PromptDialect
	RenditionRef string
}

// ModelInfo describes one catalog entry: a specific model on a specific
// lane. Field-for-field agora-spec-bridle §1. The catalog (see
// registry_models.toml + NewRegistry) is the only place ModelInfo
// values are constructed for the 7 core lanes; RegisterAlias adds
// agora-config-sourced aliases on top without minting new ModelInfo.
type ModelInfo struct {
	ID              string
	Lane            LaneID
	Provider        ProviderID
	Aliases         []string
	ContextWindow   int // tokens — required; the skills 2% catalog budget + context-manager depend on it
	MaxOutputTokens int
	Capabilities    ModelCapabilities
	Pricing         *ModelPricing    // nil = unknown/unset
	Prompt          *ModelPromptMeta // nil = no prompt-presentation metadata authored yet
}
