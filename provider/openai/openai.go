// Package openai implements the bridle Provider interface for the OpenAI API.
package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	openaiparam "github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/internal/normalize"
)

// Provider implements bridle.Provider for the OpenAI API.
type Provider struct {
	client  *openai.Client
	apiKey  string
	baseURL string

	// forceDeepSeek overrides host-based DeepSeek detection. Test seam
	// only — lets internal wire tests exercise the DeepSeek capability
	// path against a local httptest server whose host isn't
	// api.deepseek.com. Production paths leave this false and rely on
	// isDeepSeekEndpoint(baseURL).
	forceDeepSeek bool
}

// New returns an OpenAI provider.
// If apiKey is empty, the OPENAI_API_KEY environment variable is used.
func New(apiKey string) *Provider {
	return &Provider{apiKey: apiKey}
}

// NewWithBaseURL returns an OpenAI provider targeting an
// OpenAI-API-compatible endpoint at baseURL (e.g.
// "https://api.deepseek.com/v1", "https://api.together.xyz/v1",
// "http://localhost:11434/v1" for Ollama). Empty baseURL falls back
// to the SDK default (api.openai.com), matching New's behaviour.
// Useful for the many third-party services that speak the OpenAI
// Chat Completions wire shape — keeps tool / streaming / event
// plumbing while pointing requests at an alternate auth domain.
func NewWithBaseURL(apiKey, baseURL string) *Provider {
	return &Provider{apiKey: apiKey, baseURL: baseURL}
}

// NewWithClient returns an OpenAI provider using a pre-configured client.
func NewWithClient(client *openai.Client) *Provider {
	return &Provider{client: client}
}

func (p *Provider) Name() bridle.ProviderID { return bridle.ProviderOpenAI }

func (p *Provider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{
		Category:               bridle.CategoryDirectAPI,
		SupportsCustomTools:    true,
		SupportsBeforeToolCall: true,
		SupportsAfterToolCall:  true,
		SupportsMCP:            true,
	}
}

func (p *Provider) getClient() *openai.Client {
	if p.client != nil {
		return p.client
	}
	opts := make([]option.RequestOption, 0, 2)
	if p.apiKey != "" {
		opts = append(opts, option.WithAPIKey(p.apiKey))
	}
	if p.baseURL != "" {
		opts = append(opts, option.WithBaseURL(p.baseURL))
	}
	c := openai.NewClient(opts...)
	p.client = &c
	return p.client
}

// RunTurn calls the OpenAI Chat Completions API in streaming mode and
// emits bridle events to sink. Streams content deltas live so the
// funnel/UI can paint as the model speaks; uses the SDK's
// ChatCompletionAccumulator to assemble the final ChatCompletion for
// post-stream result extraction.
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	messages := toOpenAIMessages(req.AppendSystemPrompt, req.Messages)
	tools := toOpenAITools(req.Tools)

	params := openai.ChatCompletionNewParams{
		Model:    req.Model,
		Messages: messages,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	// Usage contract (NEX-581 / NEX-589): request usage on the streaming
	// response. Without stream_options.include_usage the OpenAI Chat
	// Completions streaming wire omits the usage block entirely, so the
	// accumulated completion reports zero — the hole the live A/B caught
	// on the openai-compat shim (ollama /v1) and vLLM. Setting it makes
	// the final chunk carry prompt/completion token counts, which
	// extractResult lowers into Usage. Compat servers that ignore the
	// flag still return zero; bridle's normalizeUsage floor catches
	// those at the harness layer.
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openaiparam.NewOpt(true),
	}

	// NEX-299 Pass 2: thread the sampling + output knobs the OpenAI
	// Chat Completions API exposes. Nil pointers / zero values stay
	// unset (omitzero on the SDK param).
	if req.Temperature != nil {
		params.Temperature = openaiparam.NewOpt(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = openaiparam.NewOpt(*req.TopP)
	}
	if req.Seed != nil {
		params.Seed = openaiparam.NewOpt(int64(*req.Seed))
	}
	if req.MaxOutputTokens > 0 {
		params.MaxCompletionTokens = openaiparam.NewOpt(int64(req.MaxOutputTokens))
	}
	if len(req.StopSequences) > 0 {
		// Stop accepts a string OR string array. Use the array variant
		// uniformly — single-stop callers get the same wire shape as
		// multi-stop callers, simpler than per-len branching.
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: req.StopSequences}
	}
	if rf := p.responseFormatFor(req.ResponseFormat); rf != nil {
		params.ResponseFormat = *rf
	}
	if tc := toOpenAIToolChoice(req.ToolChoice); tc != nil {
		params.ToolChoice = *tc
	}
	// TopK: claude-only; OpenAI has no equivalent. Silently ignored.

	stream := p.getClient().Chat.Completions.NewStreaming(ctx, params)
	acc := openai.ChatCompletionAccumulator{}
	// NEX-340: DeepSeek emits reasoning_content as per-chunk deltas
	// (ChatCompletionChunkChoiceDelta.JSON.ExtraFields["reasoning_content"]).
	// The SDK's ChatCompletionAccumulator folds the typed fields into
	// the final ChatCompletion but doesn't carry ExtraFields through.
	// Accumulate locally so extractResult can attach the full
	// reasoning text to the SessionDelta for cross-turn replay.
	var reasoningBuf strings.Builder
	// The accumulator folds ONLY the three top-level token totals into the
	// final ChatCompletion (streamaccumulator.go accumulateDelta) — it drops
	// prompt_tokens_details / completion_tokens_details AND every extra field
	// (the OpenRouter `cost`). So capture the usage-bearing chunk (the final
	// one, per stream_options.include_usage) verbatim here, same pattern as
	// the reasoning_content capture below; usageFromChunk lowers it after the
	// stream ends. Without this, cached/reasoning tokens and cost read zero
	// on the streaming path — which is every RunTurn.
	var streamUsage openai.CompletionUsage
	var haveStreamUsage bool
	for stream.Next() {
		chunk := stream.Current()
		if !acc.AddChunk(chunk) {
			return bridle.ProviderResult{}, fmt.Errorf("openai: accumulator rejected chunk")
		}
		if raw := chunk.Usage.RawJSON(); raw != "" && raw != "null" {
			streamUsage = chunk.Usage
			haveStreamUsage = true
		}
		// Emit content deltas live; tool-call argument deltas are
		// captured by the accumulator but not surfaced as ModelChunks.
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			sink.Emit(bridle.ModelChunk{Text: chunk.Choices[0].Delta.Content})
		}
		// Capture reasoning_content delta if present (DeepSeek reasoner).
		if len(chunk.Choices) > 0 {
			if rc, ok := chunk.Choices[0].Delta.JSON.ExtraFields["reasoning_content"]; ok {
				raw := rc.Raw()
				if raw != "" && raw != "null" {
					var piece string
					if json.Unmarshal([]byte(raw), &piece) == nil {
						reasoningBuf.WriteString(piece)
						// NEX-767 T7: stream it live too (agora-spec-bridle
						// §2 reasoning_delta) — reasoningBuf above is the
						// separate cross-turn-replay accumulation
						// (extractResult attaches it to SessionDelta); this
						// emit is the additional live surface, not a
						// replacement.
						sink.Emit(bridle.ReasoningChunk{Text: piece})
					}
				}
			}
		}
	}
	if err := stream.Err(); err != nil {
		// NEX-587: classify the API error so callers can distinguish
		// rate-limit / auth / server failures (the funnel's retry+backoff
		// keys off ProviderErrorRateLimit). Non-API errors (context
		// cancel, dial failure) classify as nil and fall through to the
		// generic wrap.
		if pe := classifyOpenAIError(err); pe != nil {
			return bridle.ProviderResult{}, pe
		}
		return bridle.ProviderResult{}, fmt.Errorf("openai: API error: %w", err)
	}

	res, err := extractResult(&acc.ChatCompletion, reasoningBuf.String())
	if err == nil && haveStreamUsage {
		// The captured chunk carries the FULL usage block straight off the
		// wire — totals, details, and extra fields — so it supersedes the
		// accumulator-derived (details-less) usage extractResult built.
		res.Usage = usageFromChunk(streamUsage)
	}
	return res, err
}

// usageFromChunk lowers a wire-verbatim CompletionUsage (captured from the
// usage-bearing stream chunk — see RunTurn) into bridle.Usage: token totals,
// the cached/reasoning detail counts, and the OpenRouter/LiteLLM `cost`
// extra field (exact upstream USD; absent on standard OpenAI backends → 0).
func usageFromChunk(u openai.CompletionUsage) bridle.Usage {
	usage := bridle.Usage{
		InputTokens:          int(u.PromptTokens),
		OutputTokens:         int(u.CompletionTokens),
		CacheReadInputTokens: int(u.PromptTokensDetails.CachedTokens),
		ReasoningTokens:      int(u.CompletionTokensDetails.ReasoningTokens),
	}
	// NB: apijson marks EXTRA fields invalid-status (they're unknown to the
	// typed schema), so Valid() is false even when a value is present — gate
	// on Raw() instead.
	if f, ok := u.JSON.ExtraFields["cost"]; ok {
		if raw := f.Raw(); raw != "" && raw != "null" {
			if v, err := strconv.ParseFloat(raw, 64); err == nil {
				usage.CostUSD = v
			}
		}
	}
	return usage
}

// extractResult pulls finalText/toolCalls/usage out of an accumulated
// ChatCompletion. Chunks were already emitted live during the stream
// loop, so this just lowers the assembled completion into a
// bridle.ProviderResult.
func extractResult(completion *openai.ChatCompletion, streamReasoning string) (bridle.ProviderResult, error) {
	if len(completion.Choices) == 0 {
		return bridle.ProviderResult{StopReason: bridle.StopReasonModelDone}, nil
	}

	choice := completion.Choices[0]
	msg := choice.Message

	var finalText string
	var toolCalls []bridle.ToolInvocation
	var sessionDelta []bridle.SessionEvent

	if msg.Content != "" {
		bridle.AppendAssistantText(&finalText, &sessionDelta, bridle.ProviderOpenAI, msg.Content)
	}

	for _, tc := range msg.ToolCalls {
		argsJSON := json.RawMessage(tc.Function.Arguments)
		toolCalls = append(toolCalls, bridle.ToolInvocation{
			ID:   tc.ID,
			Name: tc.Function.Name,
			Args: argsJSON,
		})
		raw, _ := json.Marshal(tc)
		sessionDelta = append(sessionDelta, bridle.SessionEvent{
			Provider: bridle.ProviderOpenAI,
			Role:     bridle.RoleAssistant,
			RawJSON:  raw,
		})
	}

	// NEX-340: DeepSeek reasoner-style models emit `reasoning_content`
	// per-chunk during streaming (extension to OpenAI Chat Completions
	// wire). The caller accumulates the deltas; we prefer that
	// streamed-accumulated text. As a fallback (non-streaming, or
	// providers that surface it on the final message) we also check
	// msg.JSON.ExtraFields. Attached to the FIRST assistant
	// SessionEvent so cross-Deliberate replay (lowerRequest) preserves
	// it. DeepSeek's API rejects subsequent turns whose assistant
	// history is missing the reasoning_content with 400
	// ("The reasoning_content in the thinking mode must be passed
	// back to the API"). Mirrors NEX-320 attachment pattern.
	reasoning := streamReasoning
	if reasoning == "" {
		reasoning = extractReasoningContent(msg)
	}
	if reasoning != "" {
		attached := false
		for i := range sessionDelta {
			if sessionDelta[i].Role == bridle.RoleAssistant {
				sessionDelta[i].ReasoningContent = reasoning
				attached = true
				break
			}
		}
		if !attached {
			// Reasoning-only turn (no text, no tool_use) — synthesize a
			// carrier event so the blocks survive cross-turn replay.
			sessionDelta = append([]bridle.SessionEvent{{
				Provider:         bridle.ProviderOpenAI,
				Role:             bridle.RoleAssistant,
				ReasoningContent: reasoning,
			}}, sessionDelta...)
		}
	}

	stopReason := normalize.OpenAIStopReason(string(choice.FinishReason))

	usage := bridle.Usage{
		InputTokens:  int(completion.Usage.PromptTokens),
		OutputTokens: int(completion.Usage.CompletionTokens),
		// prompt_tokens_details.cached_tokens is the prefix-cache hit count
		// on OpenAI-shape backends (OpenAI, Moonshot/kimi, DeepSeek, most
		// gateways). Lowering it makes cache behavior OBSERVABLE at the
		// host (agora's usage row) — without it the ~20%-hit placement
		// regression on kimi was invisible from inside the harness.
		// Backends that omit the field yield zero, which is also the
		// correct "no cache" report.
		CacheReadInputTokens: int(completion.Usage.PromptTokensDetails.CachedTokens),
		// completion_tokens_details.reasoning_tokens: reasoner models
		// (kimi-k3, DeepSeek) bill thinking as output; surface it so the
		// host can tell reasoning spend from answer spend.
		ReasoningTokens: int(completion.Usage.CompletionTokensDetails.ReasoningTokens),
	}
	// OpenRouter (and LiteLLM passing it through) reports the EXACT upstream
	// charge as a non-standard `cost` field (USD float) on the usage block.
	// It's not in openai-go's typed CompletionUsage, so read it from the
	// preserved raw JSON. When present it beats any host-side price-table
	// estimate; standard OpenAI backends omit it and CostUSD stays 0.
	if f, ok := completion.Usage.JSON.ExtraFields["cost"]; ok {
		if raw := f.Raw(); raw != "" && raw != "null" {
			if v, err := strconv.ParseFloat(raw, 64); err == nil {
				usage.CostUSD = v
			}
		}
	}
	return bridle.ProviderResult{
		FinalText:        finalText,
		ToolCalls:        toolCalls,
		Usage:            usage,
		StopReason:       stopReason,
		ResolvedModel:    completion.Model,
		SessionDelta:     sessionDelta,
		ReasoningContent: reasoning,
	}, nil
}

// extractReasoningContent reads DeepSeek's reasoning_content extension
// field from an openai-go ChatCompletionMessage. Returns "" when the
// field is absent or empty — non-reasoning models (vanilla OpenAI,
// DeepSeek chat/flash variants) won't have it.
//
// Note: ExtraFields entries land at status=invalid (the SDK only marks
// status=valid for typed fields it knows about), so the Valid() guard
// applied to known fields doesn't apply here — Raw() is the
// authoritative check.
func extractReasoningContent(msg openai.ChatCompletionMessage) string {
	rc, ok := msg.JSON.ExtraFields["reasoning_content"]
	if !ok {
		return ""
	}
	raw := rc.Raw()
	if raw == "" || raw == "null" {
		return ""
	}
	var out string
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return ""
	}
	return out
}

func toOpenAIMessages(systemPrompt string, msgs []bridle.ProviderMessage) []openai.ChatCompletionMessageParamUnion {
	var out []openai.ChatCompletionMessageParamUnion

	if systemPrompt != "" {
		out = append(out, openai.SystemMessage(systemPrompt))
	}

	for _, m := range msgs {
		switch m.Role {
		case "user":
			out = append(out, openai.UserMessage(m.Content))
		case "assistant":
			// OpenAI wire spec: an assistant message MUST carry either
			// content or tool_calls. reasoning_content alone is not a
			// valid body — DeepSeek rejects with "Invalid assistant
			// message: content or tool_calls must be set" 400. Drop
			// reasoning-only carriers rather than emit malformed wire;
			// without a content/tool_calls anchor there's no server
			// turn for reasoning_content to attach to in the first
			// place, so dropping it is also semantically correct.
			if m.Content == "" && len(m.ToolCalls) == 0 {
				continue
			}
			var assistant openai.ChatCompletionAssistantMessageParam
			if m.Content != "" {
				assistant.Content.OfString = openaiparam.NewOpt(m.Content)
			}
			if len(m.ToolCalls) > 0 {
				tcs := make([]openai.ChatCompletionMessageToolCallParam, 0, len(m.ToolCalls))
				for _, tc := range m.ToolCalls {
					tcs = append(tcs, openai.ChatCompletionMessageToolCallParam{
						ID: tc.ID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      tc.Name,
							Arguments: string(tc.Args),
						},
					})
				}
				assistant.ToolCalls = tcs
			}
			// NEX-340: DeepSeek requires reasoning_content round-tripped
			// on prior assistant turns. SDK doesn't have a typed field
			// for it (it's a DeepSeek-only extension to OpenAI Chat
			// Completions), so we inject via SetExtraFields. No-op for
			// non-reasoning history.
			if m.ReasoningContent != "" {
				assistant.SetExtraFields(map[string]any{
					"reasoning_content": m.ReasoningContent,
				})
			}
			out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant})
		case "tool_result":
			out = append(out, openai.ToolMessage(m.Content, m.ToolCallID))
		case "system":
			out = append(out, openai.SystemMessage(m.Content))
		}
	}
	return out
}

func toOpenAITools(defs []bridle.ToolDef) []openai.ChatCompletionToolParam {
	if len(defs) == 0 {
		return nil
	}
	out := make([]openai.ChatCompletionToolParam, 0, len(defs))
	for _, d := range defs {
		fn := shared.FunctionDefinitionParam{
			Name: d.Name,
		}
		if d.Description != "" {
			fn.Description = openai.String(d.Description)
		}
		if len(d.InputSchema) > 0 {
			var params shared.FunctionParameters
			if err := json.Unmarshal(d.InputSchema, &params); err == nil {
				fn.Parameters = params
			}
		}
		out = append(out, openai.ChatCompletionToolParam{
			Function: fn,
		})
	}
	return out
}

// toOpenAIResponseFormat maps bridle's ResponseFormat to OpenAI's
// response_format union. Nil input → nil (caller leaves the field
// unset → SDK omits → model uses free-form text default).
//
//	Type=""             → nil (free-form text default)
//	Type="text"         → ResponseFormatTextParam
//	Type="json_object"  → ResponseFormatJSONObjectParam
//	Type="json_schema"  → ResponseFormatJSONSchemaParam with strict mode
//	                      from rf.Strict. Schema must be non-empty.
//
// Unknown Type returns nil so a future-typed bridle caller doesn't
// accidentally send a malformed format to an older provider.
// NEX-299 Pass 2.
func toOpenAIResponseFormat(rf *bridle.ResponseFormat) *openai.ChatCompletionNewParamsResponseFormatUnion {
	if rf == nil {
		return nil
	}
	switch rf.Type {
	case "", "text":
		return nil // SDK default — same as not setting the field
	case "json_object":
		u := openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		}
		return &u
	case "json_schema":
		var schemaAny any
		if len(rf.Schema) > 0 {
			_ = json.Unmarshal(rf.Schema, &schemaAny)
		}
		js := shared.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:   rf.Name,
			Schema: schemaAny,
		}
		if rf.Strict {
			js.Strict = openaiparam.NewOpt(true)
		}
		if rf.Description != "" {
			js.Description = openaiparam.NewOpt(rf.Description)
		}
		u := openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{JSONSchema: js},
		}
		return &u
	default:
		return nil
	}
}

// toOpenAIToolChoice maps bridle's tool_choice string to OpenAI's
// tool_choice union.
//
//	"" → nil (provider default, usually "auto")
//	"auto" / "none" / "required" → string variant (matches OpenAI spec;
//	                               bridle's "any" maps to OpenAI's "required")
//	"<name>" → named tool variant forcing the model to call <name>
//
// NEX-299 Pass 2. Pre-fix the openai provider silently dropped
// ToolChoice entirely even though bridle.ProviderRequest carried it.
func toOpenAIToolChoice(choice string) *openai.ChatCompletionToolChoiceOptionUnionParam {
	switch choice {
	case "":
		return nil
	case "auto", "none", "required":
		u := openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openaiparam.NewOpt(choice)}
		return &u
	case "any":
		// bridle's "any" semantically == OpenAI's "required"
		u := openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openaiparam.NewOpt("required")}
		return &u
	default:
		named := openai.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
			openai.ChatCompletionNamedToolChoiceFunctionParam{Name: choice},
		)
		return &named
	}
}
