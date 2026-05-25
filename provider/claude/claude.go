// Package claude implements the bridle Provider interface for the Anthropic Claude API.
package claude

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/internal/normalize"
)

// Provider implements bridle.Provider for the Anthropic Claude API.
type Provider struct {
	client  *anthropic.Client
	apiKey  string
	baseURL string
}

// New returns a Claude provider.
// If apiKey is empty, the ANTHROPIC_API_KEY environment variable is used.
func New(apiKey string) *Provider {
	return &Provider{apiKey: apiKey}
}

// NewWithBaseURL returns a Claude provider targeting an
// Anthropic-API-compatible endpoint at baseURL (e.g.
// "https://api.deepseek.com/anthropic"). Empty baseURL falls back to
// the SDK default (api.anthropic.com), matching New's behaviour.
// Useful for third-party services that speak the Anthropic Messages
// wire shape — keeps the existing tool / streaming / event plumbing
// while pointing requests at an alternate auth domain.
func NewWithBaseURL(apiKey, baseURL string) *Provider {
	return &Provider{apiKey: apiKey, baseURL: baseURL}
}

// NewWithClient returns a Claude provider using a pre-configured client.
func NewWithClient(client *anthropic.Client) *Provider {
	return &Provider{client: client}
}

func (p *Provider) Name() bridle.ProviderID { return bridle.ProviderClaude }

func (p *Provider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{
		Category:               bridle.CategoryDirectAPI,
		SupportsCustomTools:    true,
		SupportsBeforeToolCall: true,
		SupportsAfterToolCall:  true,
		SupportsMCP:            true,
	}
}

func (p *Provider) getClient() *anthropic.Client {
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
	c := anthropic.NewClient(opts...)
	p.client = &c
	return p.client
}

// RunTurn calls the Claude Messages API in streaming mode and emits
// bridle events to sink. Streams TextDelta events live so the funnel/UI
// can paint as the model speaks; accumulates into a Message via the
// SDK's Accumulate helper and extracts the final result post-stream.
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	messages, err := toClaudeMessages(req.Messages)
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("claude: message conversion: %w", err)
	}

	// AppendSystemPrompt is the field name shared across all bridle providers
	// for consistency with claudecode (where the semantic distinction matters —
	// see provider/claudecode/claudecode.go for why). Anthropic's chat
	// completions API has no "default system prompt" injected by the runtime;
	// the caller's content IS the whole system message. So here the field
	// functions identically to a plain SystemPrompt — rename is semantic
	// consistency, not behavior change.
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		System:    toClaudeSystem(req.AppendSystemPrompt),
		Messages:  messages,
		MaxTokens: 4096,
	}

	tools := toClaudeTools(req.Tools)
	if len(tools) > 0 {
		params.Tools = tools
	}

	stream := p.getClient().Messages.NewStreaming(ctx, params)
	var msg anthropic.Message
	for stream.Next() {
		event := stream.Current()
		if accErr := msg.Accumulate(event); accErr != nil {
			return bridle.ProviderResult{}, fmt.Errorf("claude: accumulate stream event: %w", accErr)
		}
		// Emit TextDelta chunks live so callers can paint as they arrive.
		// Other delta types (InputJSONDelta for tool args, etc.) are
		// captured by Accumulate but not surfaced as ModelChunks — those
		// would be misinterpreted as model prose.
		if delta, ok := event.AsAny().(anthropic.ContentBlockDeltaEvent); ok {
			if td, ok := delta.Delta.AsAny().(anthropic.TextDelta); ok {
				sink.Emit(bridle.ModelChunk{Text: td.Text})
			}
		}
	}
	if err := stream.Err(); err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("claude: API error: %w", err)
	}

	return extractResult(&msg)
}

// extractResult pulls finalText/toolCalls/usage out of an accumulated
// Anthropic Message. Chunks were already emitted live during the
// stream loop, so this just lowers the assembled message into a
// bridle.ProviderResult.
func extractResult(msg *anthropic.Message) (bridle.ProviderResult, error) {
	var finalText string
	var toolCalls []bridle.ToolInvocation
	var sessionDelta []bridle.SessionEvent

	for _, block := range msg.Content {
		switch b := block.AsAny().(type) {
		case anthropic.TextBlock:
			bridle.AppendAssistantText(&finalText, &sessionDelta, bridle.ProviderClaude, b.Text)

		case anthropic.ToolUseBlock:
			toolCalls = append(toolCalls, bridle.ToolInvocation{
				ID:   b.ID,
				Name: b.Name,
				Args: b.Input,
			})
			raw, _ := json.Marshal(b)
			sessionDelta = append(sessionDelta, bridle.SessionEvent{
				Provider: bridle.ProviderClaude,
				Role:     bridle.RoleAssistant,
				RawJSON:  raw,
			})
		}
	}

	stopReason := normalize.ClaudeStopReason(string(msg.StopReason))

	return bridle.ProviderResult{
		FinalText: finalText,
		ToolCalls: toolCalls,
		Usage: bridle.Usage{
			InputTokens:              int(msg.Usage.InputTokens),
			OutputTokens:             int(msg.Usage.OutputTokens),
			CacheReadInputTokens:     int(msg.Usage.CacheReadInputTokens),
			CacheCreationInputTokens: int(msg.Usage.CacheCreationInputTokens),
		},
		StopReason:    stopReason,
		ResolvedModel: string(msg.Model),
		SessionDelta:  sessionDelta,
	}, nil
}

func toClaudeMessages(msgs []bridle.ProviderMessage) ([]anthropic.MessageParam, error) {
	var out []anthropic.MessageParam
	for _, m := range msgs {
		switch m.Role {
		case "user":
			out = append(out, anthropic.NewUserMessage(
				anthropic.NewTextBlock(m.Content),
			))
		case "assistant":
			blocks := []anthropic.ContentBlockParamUnion{}
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, tc.Args, tc.Name))
			}
			if len(blocks) == 0 {
				// Empty assistant turn — skip rather than emit invalid empty content
				continue
			}
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		case "tool_result":
			out = append(out, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(m.ToolCallID, m.Content, false),
			))
		case "system":
			// System tail events folded into a user message as context.
			out = append(out, anthropic.NewUserMessage(
				anthropic.NewTextBlock("[system context] "+m.Content),
			))
		}
	}
	return out, nil
}

func toClaudeSystem(prompt string) []anthropic.TextBlockParam {
	if prompt == "" {
		return nil
	}
	return []anthropic.TextBlockParam{{Text: prompt}}
}

// jsonSchemaShape mirrors the subset of JSON Schema the Anthropic
// ToolInputSchemaParam consumes. Only `properties` and `required`
// are surfaced as typed fields; `type` is always "object" and the
// SDK default handles it. Anything else (description, default,
// enum on nested properties, $schema/$id metadata, etc.) is
// preserved verbatim under `properties` — JSON Schema's recursive
// shape means we don't need to walk it ourselves.
type jsonSchemaShape struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Required   []string       `json:"required"`
}

// toClaudeTools converts bridle's ToolDef list to the Anthropic
// Messages tool spec. The InputSchema is parsed as JSON Schema and
// destructured into ToolInputSchemaParam's typed fields.
//
// Pre-fix (NEX-299 Pass 1, 2026-05-26): the previous implementation
// unmarshalled the entire schema object into ToolInputSchemaParam.
// Properties, producing a wire payload like
//
//	{"properties": {"type": "object",
//	                "properties": {"text": {...}},
//	                "required": ["text"]}}
//
// — structurally wrong (properties nested inside properties). Real
// api.anthropic.com tolerated this with lenient parsing; strict
// validators (DeepSeek's /anthropic translation gateway) reject it
// with "Invalid schema for function 'foo': [\"x\"] is not of types
// \"boolean\", \"object\"" because the misplaced `required` array
// looks like a malformed property entry. Found via nexus
// test-provider --tools against DeepSeek (NEX-297 L1).
//
// The fix lifts `properties` and `required` to their proper top-level
// positions on ToolInputSchemaParam so the wire payload matches the
// Anthropic spec.
func toClaudeTools(defs []bridle.ToolDef) []anthropic.ToolUnionParam {
	out := make([]anthropic.ToolUnionParam, 0, len(defs))
	for _, d := range defs {
		schema := anthropic.ToolInputSchemaParam{}
		if len(d.InputSchema) > 0 {
			var parsed jsonSchemaShape
			if err := json.Unmarshal(d.InputSchema, &parsed); err == nil {
				schema.Properties = parsed.Properties
				schema.Required = parsed.Required
			}
		}
		out = append(out, anthropic.ToolUnionParamOfTool(schema, d.Name))
		// Description is on ToolParam, set via the variant directly.
		if d.Description != "" && len(out) > 0 {
			if out[len(out)-1].OfTool != nil {
				out[len(out)-1].OfTool.Description = anthropic.String(d.Description)
			}
		}
	}
	return out
}
