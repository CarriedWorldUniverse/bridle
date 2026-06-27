// Package ollama implements the bridle Provider interface for a local Ollama server.
package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/api"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/internal/normalize"
)

const defaultBaseURL = "http://localhost:11434"

// defaultKeepAlive holds gemma-class models loaded across quiet
// periods: ollama's server-side default unloads after ~5 idle minutes,
// which makes an always-on aspect (keel) pay a full model reload on
// the first turn after every lull.
const defaultKeepAlive = 30 * time.Minute

// Provider implements bridle.Provider for a local Ollama server.
type Provider struct {
	client  *api.Client
	baseURL string

	// KeepAlive controls how long the server keeps the model loaded
	// after a request. 0 means the 30m default (keel is always-on).
	KeepAlive time.Duration
	// NumCtx sets the context window (options.num_ctx). 0 means omit,
	// leaving the model default in effect.
	NumCtx int
	// Options is merged into ChatRequest.Options on every turn;
	// NumCtx wins on conflict. The map itself is never mutated.
	Options map[string]any
}

// New returns an Ollama provider pointing at the default localhost:11434.
func New() *Provider {
	return &Provider{baseURL: defaultBaseURL}
}

// NewWithURL returns an Ollama provider pointing at a custom server URL.
func NewWithURL(baseURL string) *Provider {
	return &Provider{baseURL: baseURL}
}

// NewWithClient returns an Ollama provider using a pre-configured client.
func NewWithClient(client *api.Client) *Provider {
	return &Provider{client: client}
}

func (p *Provider) Name() bridle.ProviderID { return bridle.ProviderOllama }

func (p *Provider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{
		Category:               bridle.CategoryDirectAPI,
		SupportsCustomTools:    true,
		SupportsBeforeToolCall: true,
		SupportsAfterToolCall:  true,
		SupportsMCP:            true,
	}
}

func (p *Provider) getClient() (*api.Client, error) {
	if p.client != nil {
		return p.client, nil
	}
	u, err := url.Parse(p.baseURL)
	if err != nil {
		return nil, fmt.Errorf("ollama: invalid base URL %q: %w", p.baseURL, err)
	}
	client := api.NewClient(u, http.DefaultClient)
	p.client = client
	return client, nil
}

// RunTurn calls the Ollama Chat API and emits bridle events to sink.
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	client, err := p.getClient()
	if err != nil {
		return bridle.ProviderResult{}, err
	}

	messages := toOllamaMessages(req.Messages)
	tools := toOllamaTools(req.Tools)

	options := make(map[string]any, len(p.Options)+1)
	for k, v := range p.Options {
		options[k] = v
	}
	applyContextPolicy(options, req.ContextPolicy, p.NumCtx)

	ka := p.KeepAlive
	if ka == 0 {
		ka = defaultKeepAlive
	}

	stream := true
	chatReq := &api.ChatRequest{
		Model:     req.Model,
		Messages:  messages,
		Tools:     tools,
		Stream:    &stream,
		Options:   options,
		KeepAlive: &api.Duration{Duration: ka},
	}
	if req.AppendSystemPrompt != "" {
		chatReq.Messages = append([]api.Message{
			{Role: "system", Content: req.AppendSystemPrompt},
		}, chatReq.Messages...)
	}

	// In streaming mode the callback fires once per chunk: each
	// Message.Content carries only the new delta, not the full text.
	// We emit deltas live, accumulate the full text ourselves, and use
	// the final (Done=true) callback for tool_calls + done_reason.
	var (
		finalResp     api.ChatResponse
		accumulatedTx strings.Builder
	)
	err = client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
		finalResp = resp
		if resp.Message.Content != "" {
			sink.Emit(bridle.ModelChunk{Text: resp.Message.Content})
			accumulatedTx.WriteString(resp.Message.Content)
		}
		return nil
	})
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("ollama: Chat error: %w", err)
	}

	// Overwrite the (delta-only) Content on the final response with the
	// accumulated text before lowering — extractResult reads it as the
	// model's full reply.
	finalResp.Message.Content = accumulatedTx.String()
	return extractResult(finalResp), nil
}

// applyContextPolicy maps bridle's context contract (NEX-581) to
// ollama's engine knob: options.num_ctx. The per-request
// policy.TargetWindow wins when set; otherwise the static
// Provider.NumCtx holds; otherwise num_ctx is omitted entirely so the
// model default is in effect. PromptBudget is engine-agnostic and
// enforced at the harness seam, so it has no effect here.
//
// The chosen num_ctx is written into options (overwriting any num_ctx a
// caller passed via Provider.Options — the explicit window policy wins
// over a passthrough option, matching the documented NumCtx precedence).
func applyContextPolicy(options map[string]any, policy bridle.ContextPolicy, staticNumCtx int) {
	numCtx := staticNumCtx
	if policy.TargetWindow > 0 {
		numCtx = policy.TargetWindow
	}
	if numCtx > 0 {
		options["num_ctx"] = numCtx
	}
}

func extractResult(resp api.ChatResponse) bridle.ProviderResult {
	var toolCalls []bridle.ToolInvocation
	var sessionDelta []bridle.SessionEvent

	if resp.Message.Content != "" {
		sessionDelta = append(sessionDelta, bridle.SessionEvent{
			Provider: bridle.ProviderOllama,
			Role:     bridle.RoleAssistant,
			Content:  resp.Message.Content,
		})
	}

	for _, tc := range resp.Message.ToolCalls {
		argsJSON, _ := json.Marshal(tc.Function.Arguments)
		id := tc.ID
		if id == "" {
			id = tc.Function.Name
		}
		toolCalls = append(toolCalls, bridle.ToolInvocation{
			ID:   id,
			Name: tc.Function.Name,
			Args: argsJSON,
		})
		raw, _ := json.Marshal(tc)
		sessionDelta = append(sessionDelta, bridle.SessionEvent{
			Provider: bridle.ProviderOllama,
			Role:     bridle.RoleAssistant,
			RawJSON:  raw,
		})
	}

	stopReason := normalize.OllamaStopReason(resp.DoneReason)

	return bridle.ProviderResult{
		FinalText: resp.Message.Content,
		ToolCalls: toolCalls,
		Usage: bridle.Usage{
			InputTokens:  resp.PromptEvalCount,
			OutputTokens: resp.EvalCount,
		},
		StopReason:    stopReason,
		ResolvedModel: resp.Model,
		SessionDelta:  sessionDelta,
	}
}

func toOllamaMessages(msgs []bridle.ProviderMessage) []api.Message {
	out := make([]api.Message, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case "user", "system":
			out = append(out, api.Message{Role: m.Role, Content: m.Content})
		case "assistant":
			msg := api.Message{Role: "assistant", Content: m.Content}
			for _, tc := range m.ToolCalls {
				var args api.ToolCallFunctionArguments
				if len(tc.Args) > 0 {
					_ = args.UnmarshalJSON(tc.Args)
				}
				msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
					ID: tc.ID,
					Function: api.ToolCallFunction{
						Name:      tc.Name,
						Arguments: args,
					},
				})
			}
			if msg.Content == "" && len(msg.ToolCalls) == 0 {
				continue
			}
			out = append(out, msg)
		case "tool_result":
			out = append(out, api.Message{
				Role:       "tool",
				Content:    m.Content,
				ToolCallID: m.ToolCallID,
			})
		}
	}
	return out
}

func toOllamaTools(defs []bridle.ToolDef) api.Tools {
	if len(defs) == 0 {
		return nil
	}
	tools := make(api.Tools, 0, len(defs))
	for _, d := range defs {
		fn := api.ToolFunction{
			Name:        d.Name,
			Description: d.Description,
		}
		if len(d.InputSchema) > 0 {
			var schema map[string]interface{}
			if err := json.Unmarshal(d.InputSchema, &schema); err == nil {
				if props, ok := schema["properties"]; ok {
					propsJSON, _ := json.Marshal(props)
					var propsMap map[string]api.ToolProperty
					if err := json.Unmarshal(propsJSON, &propsMap); err == nil {
						pm := api.NewToolPropertiesMap()
						for k, v := range propsMap {
							pm.Set(k, v)
						}
						fn.Parameters.Properties = pm
					}
				}
				if req, ok := schema["required"].([]interface{}); ok {
					for _, r := range req {
						if s, ok := r.(string); ok {
							fn.Parameters.Required = append(fn.Parameters.Required, s)
						}
					}
				}
			}
		}
		tools = append(tools, api.Tool{
			Type:     "function",
			Function: fn,
		})
	}
	return tools
}
