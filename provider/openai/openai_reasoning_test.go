package openai

import (
	"encoding/json"
	"strings"
	"testing"

	openaisdk "github.com/openai/openai-go"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// NEX-340: extractReasoningContent pulls DeepSeek's reasoning_content
// extension field off the assistant message. The openai-go SDK
// surfaces unknown fields via msg.JSON.ExtraFields.
func TestExtractReasoningContent_DeepSeekExtension(t *testing.T) {
	// Build a ChatCompletionMessage by unmarshaling the wire shape
	// DeepSeek emits — bypasses the typed-field constructors that
	// would refuse to set ExtraFields. The respjson layer captures
	// "reasoning_content" as an ExtraField when it's not in the SDK's
	// typed struct.
	body := []byte(`{
		"role": "assistant",
		"content": "Forty-two.",
		"refusal": "",
		"reasoning_content": "Walked through 6*7 step by step."
	}`)
	var msg openaisdk.ChatCompletionMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got := extractReasoningContent(msg)
	if got != "Walked through 6*7 step by step." {
		t.Errorf("got %q, want the reasoning text", got)
	}
}

func TestExtractReasoningContent_AbsentReturnsEmpty(t *testing.T) {
	body := []byte(`{"role":"assistant","content":"hi","refusal":""}`)
	var msg openaisdk.ChatCompletionMessage
	_ = json.Unmarshal(body, &msg)
	if got := extractReasoningContent(msg); got != "" {
		t.Errorf("got %q, want empty for non-reasoning model", got)
	}
}

func TestExtractReasoningContent_NullReturnsEmpty(t *testing.T) {
	body := []byte(`{"role":"assistant","content":"hi","refusal":"","reasoning_content":null}`)
	var msg openaisdk.ChatCompletionMessage
	_ = json.Unmarshal(body, &msg)
	if got := extractReasoningContent(msg); got != "" {
		t.Errorf("got %q, want empty for null reasoning_content", got)
	}
}

// NEX-340 cross-turn: toOpenAIMessages emits reasoning_content on
// assistant messages that carry it, via SetExtraFields. Verifies the
// JSON wire shape DeepSeek will accept on the next turn.
func TestToOpenAIMessages_EmitsReasoningContentOnAssistant(t *testing.T) {
	msgs := []bridle.ProviderMessage{
		{Role: "user", Content: "what's 6*7?"},
		{Role: "assistant", Content: "Forty-two.",
			ReasoningContent: "Walked through 6*7 step by step."},
		{Role: "user", Content: "and 7*8?"},
	}
	out := toOpenAIMessages("", msgs)
	if len(out) != 3 {
		t.Fatalf("got %d messages, want 3", len(out))
	}
	// Marshal the assistant param and confirm reasoning_content lands.
	assistantBytes, err := out[1].MarshalJSON()
	if err != nil {
		t.Fatalf("marshal assistant: %v", err)
	}
	body := string(assistantBytes)
	if !strings.Contains(body, `"reasoning_content"`) {
		t.Errorf("assistant message missing reasoning_content key:\n%s", body)
	}
	if !strings.Contains(body, "Walked through 6*7 step by step.") {
		t.Errorf("assistant message missing reasoning text:\n%s", body)
	}
}

// Assistant messages WITHOUT reasoning_content (vanilla OpenAI, non-
// reasoning DeepSeek) don't get the extra field — wire stays
// backward-compatible.
func TestToOpenAIMessages_OmitsReasoningContentWhenAbsent(t *testing.T) {
	msgs := []bridle.ProviderMessage{
		{Role: "assistant", Content: "hi"},
	}
	out := toOpenAIMessages("", msgs)
	if len(out) != 1 {
		t.Fatalf("got %d", len(out))
	}
	body, _ := out[0].MarshalJSON()
	if strings.Contains(string(body), "reasoning_content") {
		t.Errorf("assistant message should not carry reasoning_content key when empty:\n%s", body)
	}
}
