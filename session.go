package bridle

import (
	"encoding/json"
	"fmt"
)

// SessionRole identifies who produced a session event.
type SessionRole string

const (
	RoleUser      SessionRole = "user"
	RoleAssistant SessionRole = "assistant"
	RoleTool      SessionRole = "tool"
	RoleSystem    SessionRole = "system"
)

// NormalizedSessionEvent is a provider-agnostic view of a SessionEvent,
// used when displaying or processing session history without knowing the
// provider's wire format.
type NormalizedSessionEvent struct {
	Role    SessionRole
	Content string // human-readable text representation
}

// ParseSessionEvent returns a normalized view of a session event.
// For events with RawJSON, it attempts a provider-specific parse;
// falls back to Content if RawJSON is absent or unrecognized.
func ParseSessionEvent(e SessionEvent) (NormalizedSessionEvent, error) {
	if len(e.RawJSON) == 0 || e.Content != "" {
		return NormalizedSessionEvent{Role: e.Role, Content: e.Content}, nil
	}
	// Provider-specific RawJSON parsing.
	switch e.Provider {
	case ProviderClaude, ProviderClaudeCode:
		// Both Anthropic providers use the same block shape.
		var block struct {
			Type  string          `json:"type"`
			Text  string          `json:"text"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(e.RawJSON, &block); err == nil {
			switch block.Type {
			case "text":
				return NormalizedSessionEvent{Role: e.Role, Content: block.Text}, nil
			case "tool_use":
				return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("tool_use: %s %s", block.Name, block.Input)}, nil
			}
		}
	case ProviderOllama:
		var tc struct {
			Function struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		}
		if err := json.Unmarshal(e.RawJSON, &tc); err == nil && tc.Function.Name != "" {
			return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("tool_use: %s %s", tc.Function.Name, tc.Function.Arguments)}, nil
		}
	case ProviderOpenAI:
		var tc struct {
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		}
		if err := json.Unmarshal(e.RawJSON, &tc); err == nil && tc.Function.Name != "" {
			return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("tool_use: %s %s", tc.Function.Name, tc.Function.Arguments)}, nil
		}
	case ProviderBedrock:
		var tu struct {
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(e.RawJSON, &tu); err == nil && tu.Name != "" {
			return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("tool_use: %s %s", tu.Name, tu.Input)}, nil
		}
	case ProviderGeminiCLI:
		var ev struct {
			Type       string          `json:"type"`
			ToolName   string          `json:"tool_name"`
			Parameters json.RawMessage `json:"parameters"`
			SessionID  string          `json:"session_id"`
			Model      string          `json:"model"`
		}
		if err := json.Unmarshal(e.RawJSON, &ev); err == nil {
			switch ev.Type {
			case "tool_use":
				return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("tool_use: %s %s", ev.ToolName, ev.Parameters)}, nil
			case "init", "":
				if ev.SessionID != "" {
					return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("init: session=%s model=%s", ev.SessionID, ev.Model)}, nil
				}
			}
		}
	case ProviderCodexCLI:
		var ev struct {
			Type     string `json:"type"`
			ThreadID string `json:"thread_id"`
			Command  string `json:"command"`
		}
		if err := json.Unmarshal(e.RawJSON, &ev); err == nil {
			switch ev.Type {
			case "thread.started":
				return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("init: thread=%s", ev.ThreadID)}, nil
			case "command_execution":
				return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("tool_use: command_execution %s", ev.Command)}, nil
			}
		}
	case ProviderGemini:
		var fc struct {
			FunctionCall struct {
				Name string          `json:"name"`
				Args json.RawMessage `json:"args"`
			} `json:"functionCall"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(e.RawJSON, &fc); err == nil {
			if fc.FunctionCall.Name != "" {
				return NormalizedSessionEvent{Role: e.Role, Content: fmt.Sprintf("tool_use: %s %s", fc.FunctionCall.Name, fc.FunctionCall.Args)}, nil
			}
			if fc.Text != "" {
				return NormalizedSessionEvent{Role: e.Role, Content: fc.Text}, nil
			}
		}
	}
	// Unknown provider or shape — return raw bytes as content.
	return NormalizedSessionEvent{Role: e.Role, Content: string(e.RawJSON)}, nil
}

// SessionHandle is an opaque reference to provider-side session state.
// The funnel mints handles and maps them to threads; the provider uses the
// ID to resume state (e.g., --resume <session-id> for subprocess-stream).
// For direct-api providers, Handle may be empty; state comes from SessionTail.
//
// New tells the provider whether the funnel is initiating a fresh session
// for this ID (true) or asking it to continue an existing one (false).
// For subprocess-stream providers like claudecode that maintain their own
// jsonl files, this is the difference between "create with this id" vs
// "load existing id". Direct-api providers that derive state from
// SessionTail can ignore this field.
type SessionHandle struct {
	ID  string // opaque to the funnel; meaningful to the provider
	New bool   // true on the first invocation for this ID; false on continuations
}

// SessionEvent is a single entry in a session's event log.
// The harness consumes SessionTail on the way in and proposes SessionDelta
// on the way out.
type SessionEvent struct {
	Provider ProviderID  `json:"provider,omitempty"` // who produced this event
	Role     SessionRole `json:"role"`
	Content  string      `json:"content,omitempty"`
	// RawJSON carries provider-specific blocks (tool_use, tool_result, etc.)
	// that don't fit the plain content field. Valid only in conjunction with Provider.
	RawJSON json.RawMessage `json:"raw,omitempty"`
	// ThinkingBlocks (NEX-320 cross-turn) preserves Claude extended-
	// thinking blocks across Deliberate calls — within a single Run the
	// blocks survive via ProviderResult.ThinkingBlocks, but the funnel's
	// SessionTail re-lowering path (run.go lowerRequest) erases them
	// without this field. Anthropic API rejects multi-turn requests
	// whose assistant history is missing thinking blocks from prior
	// thinking-mode turns ("content[].thinking ... must be passed back").
	//
	// Attached to the FIRST assistant SessionEvent of each turn (text
	// or tool_use) — duplicating across all events in the same turn
	// would double-emit the blocks on the wire. lowerRequest carries
	// these onto the corresponding ProviderMessage.
	//
	// omitempty keeps existing JSONL session logs backward-compatible
	// (old entries decode with empty slice; cross-turn replay no-ops
	// when nil).
	ThinkingBlocks []ThinkingBlock `json:"thinking_blocks,omitempty"`

	// ReasoningContent (NEX-340) is the openai-shape parallel of
	// ThinkingBlocks for DeepSeek reasoner-style models. Same shape
	// of requirement: subsequent turns whose assistant history loses
	// the reasoning_content field get rejected with 400
	// ("The reasoning_content in the thinking mode must be passed
	// back to the API"). Attached to the FIRST assistant SessionEvent
	// of each turn so lowerRequest can re-emit on the corresponding
	// ProviderMessage. omitempty preserves back-compat for non-
	// reasoning sessions and non-openai-shape providers.
	ReasoningContent string `json:"reasoning_content,omitempty"`

	// ToolCallID links a tool-result SessionEvent (Role==RoleTool)
	// back to the assistant tool_use that produced it. Required for
	// cross-turn replay through SessionTail: providers that use call-
	// id correlation (Anthropic, OpenAI, Ollama) need this on the
	// reconstructed tool_result ProviderMessage. Empty for non-tool
	// events. Without it, lowerRequest can't pair tool_results with
	// their tool_calls and the API rejects the rebuilt history.
	ToolCallID string `json:"tool_call_id,omitempty"`
}
