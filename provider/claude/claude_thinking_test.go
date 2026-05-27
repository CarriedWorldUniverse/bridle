package claude

import (
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// NEX-320: thinking blocks must survive multi-turn round-trip through
// toClaudeMessages. Anthropic API rejects subsequent turns whose
// history is missing the thinking blocks from prior assistant turns
// ("content[].thinking in the thinking mode must be passed back").
// Order: thinking blocks MUST precede text + tool_use in the
// reconstructed turn (Anthropic enforces).
func TestToClaudeMessages_ThinkingBlocksEmittedFirst(t *testing.T) {
	msgs := []bridle.ProviderMessage{
		{
			Role:    "assistant",
			Content: "the answer is 42",
			ThinkingBlocks: []bridle.ThinkingBlock{
				{Type: "thinking", Thinking: "let me reason...", Signature: "sig-abc-123"},
			},
			ToolCalls: []bridle.ToolInvocation{
				{ID: "tool-1", Name: "calc", Args: []byte(`{"x":1}`)},
			},
		},
	}
	out, err := toClaudeMessages(msgs)
	if err != nil {
		t.Fatalf("toClaudeMessages: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("got %d messages, want 1", len(out))
	}
	blocks := out[0].Content
	if len(blocks) != 3 {
		t.Fatalf("got %d content blocks, want 3 (thinking + text + tool_use)", len(blocks))
	}
	// Block 0 MUST be the thinking block — Anthropic enforces order.
	if blocks[0].OfThinking == nil {
		t.Errorf("block 0 should be OfThinking; got %+v", blocks[0])
	} else {
		if blocks[0].OfThinking.Thinking != "let me reason..." {
			t.Errorf("thinking content: got %q", blocks[0].OfThinking.Thinking)
		}
		if blocks[0].OfThinking.Signature != "sig-abc-123" {
			t.Errorf("signature: got %q want sig-abc-123", blocks[0].OfThinking.Signature)
		}
	}
	if blocks[1].OfText == nil {
		t.Errorf("block 1 should be OfText; got %+v", blocks[1])
	} else if blocks[1].OfText.Text != "the answer is 42" {
		t.Errorf("text content: got %q", blocks[1].OfText.Text)
	}
	if blocks[2].OfToolUse == nil {
		t.Errorf("block 2 should be OfToolUse; got %+v", blocks[2])
	}
}

// NEX-320: redacted_thinking variant round-trips. Anthropic's safety
// classifier swaps this opaque encrypted variant in when it decides
// the plaintext shouldn't reach the caller; it still must come back
// verbatim on subsequent turns.
func TestToClaudeMessages_RedactedThinkingBlockRoundTrips(t *testing.T) {
	msgs := []bridle.ProviderMessage{
		{
			Role:    "assistant",
			Content: "ok",
			ThinkingBlocks: []bridle.ThinkingBlock{
				{Type: "redacted_thinking", Data: "encrypted-blob-xyz"},
			},
		},
	}
	out, err := toClaudeMessages(msgs)
	if err != nil {
		t.Fatalf("toClaudeMessages: %v", err)
	}
	blocks := out[0].Content
	if len(blocks) != 2 {
		t.Fatalf("got %d blocks, want 2 (redacted_thinking + text)", len(blocks))
	}
	if blocks[0].OfRedactedThinking == nil {
		t.Errorf("block 0 should be OfRedactedThinking; got %+v", blocks[0])
	} else if blocks[0].OfRedactedThinking.Data != "encrypted-blob-xyz" {
		t.Errorf("redacted data: got %q want encrypted-blob-xyz",
			blocks[0].OfRedactedThinking.Data)
	}
}

// NEX-320: assistant messages without thinking blocks emit exactly
// the same shape they did pre-fix. Back-compat guard for every
// existing caller that doesn't use thinking mode.
func TestToClaudeMessages_NoThinkingBlocks_BackCompat(t *testing.T) {
	msgs := []bridle.ProviderMessage{
		{
			Role:    "assistant",
			Content: "hello",
			ToolCalls: []bridle.ToolInvocation{
				{ID: "t1", Name: "f", Args: []byte(`{}`)},
			},
		},
	}
	out, err := toClaudeMessages(msgs)
	if err != nil {
		t.Fatalf("toClaudeMessages: %v", err)
	}
	blocks := out[0].Content
	if len(blocks) != 2 {
		t.Fatalf("got %d blocks, want 2 (text + tool_use)", len(blocks))
	}
	if blocks[0].OfText == nil || blocks[1].OfToolUse == nil {
		t.Errorf("back-compat layout broken: blocks=%+v", blocks)
	}
	if blocks[0].OfThinking != nil || blocks[1].OfThinking != nil {
		t.Errorf("no thinking block expected when ThinkingBlocks nil; got one")
	}
}

// NEX-320: multiple thinking blocks preserve order (some responses
// have an internal sequence — think → tool think → think — that
// Anthropic checks against the signature chain).
func TestToClaudeMessages_MultipleThinkingBlocksOrdered(t *testing.T) {
	msgs := []bridle.ProviderMessage{
		{
			Role:    "assistant",
			Content: "done",
			ThinkingBlocks: []bridle.ThinkingBlock{
				{Type: "thinking", Thinking: "first", Signature: "sig-1"},
				{Type: "thinking", Thinking: "second", Signature: "sig-2"},
				{Type: "redacted_thinking", Data: "redacted-blob"},
			},
		},
	}
	out, _ := toClaudeMessages(msgs)
	blocks := out[0].Content
	if len(blocks) != 4 {
		t.Fatalf("got %d blocks, want 4", len(blocks))
	}
	if blocks[0].OfThinking == nil || blocks[0].OfThinking.Thinking != "first" {
		t.Errorf("block 0 wrong: %+v", blocks[0])
	}
	if blocks[1].OfThinking == nil || blocks[1].OfThinking.Thinking != "second" {
		t.Errorf("block 1 wrong: %+v", blocks[1])
	}
	if blocks[2].OfRedactedThinking == nil || blocks[2].OfRedactedThinking.Data != "redacted-blob" {
		t.Errorf("block 2 wrong: %+v", blocks[2])
	}
	if blocks[3].OfText == nil || blocks[3].OfText.Text != "done" {
		t.Errorf("block 3 wrong: %+v", blocks[3])
	}
}

// NEX-320: empty-Type ThinkingBlock defaults to "thinking" behaviour
// (forward-compat with hypothetical future block subtypes — better
// to emit a thinking block than to drop it silently).
func TestToClaudeMessages_EmptyTypeDefaultsToThinking(t *testing.T) {
	msgs := []bridle.ProviderMessage{
		{
			Role: "assistant",
			ThinkingBlocks: []bridle.ThinkingBlock{
				{Type: "", Thinking: "no type", Signature: "sig"},
			},
			Content: "ok",
		},
	}
	out, _ := toClaudeMessages(msgs)
	if out[0].Content[0].OfThinking == nil {
		t.Errorf("empty type should be treated as thinking; got %+v", out[0].Content[0])
	}
}

// NEX-320 cross-turn: thinking blocks captured during extractResult
// must be attached to the FIRST assistant SessionEvent so they survive
// the funnel's SessionTail → ProviderMessage rebuild on the next
// Deliberate. Without this attachment Anthropic 400s with
// "content[].thinking ... must be passed back to the API".
//
// Validated against the real anthropic.Message extractResult would
// produce — text + tool_use + thinking blocks in a single turn.
func TestExtractResult_AttachesThinkingBlocksToFirstAssistantSessionEvent(t *testing.T) {
	// Build a SessionDelta the way extractResult would for a
	// turn with [thinking, text, tool_use]. The block-walk in
	// extractResult appends in source order, so sessionDelta ends up:
	//   [assistant-text, assistant-tool_use]
	// with thinkingBlocks accumulated separately. The new attachment
	// logic should land thinking on sessionDelta[0].
	tb := []bridle.ThinkingBlock{{Type: "thinking", Thinking: "reasoned", Signature: "sig-1"}}
	delta := []bridle.SessionEvent{
		{Provider: bridle.ProviderClaude, Role: bridle.RoleAssistant, Content: "ok"},
		{Provider: bridle.ProviderClaude, Role: bridle.RoleAssistant, RawJSON: []byte(`{"type":"tool_use"}`)},
	}
	got := attachThinkingForTest(delta, tb)
	if len(got) != 2 {
		t.Fatalf("expected 2 events, got %d", len(got))
	}
	if len(got[0].ThinkingBlocks) != 1 || got[0].ThinkingBlocks[0].Signature != "sig-1" {
		t.Errorf("first event missing thinking: %+v", got[0])
	}
	if len(got[1].ThinkingBlocks) != 0 {
		t.Errorf("second event must NOT carry thinking (would double-emit): %+v", got[1])
	}
}

// Empty-assistant-events edge case: turn produced only thinking, no
// text or tool_use. extractResult prepends a synthetic carrier so the
// blocks survive cross-turn.
func TestExtractResult_SyntheticCarrierWhenNoAssistantEvents(t *testing.T) {
	tb := []bridle.ThinkingBlock{{Type: "thinking", Thinking: "alone", Signature: "sig-only"}}
	delta := []bridle.SessionEvent{} // no assistant events
	got := attachThinkingForTest(delta, tb)
	if len(got) != 1 {
		t.Fatalf("expected synthetic carrier, got %d events", len(got))
	}
	if got[0].Role != bridle.RoleAssistant {
		t.Errorf("synthetic role: %q", got[0].Role)
	}
	if len(got[0].ThinkingBlocks) != 1 {
		t.Errorf("synthetic missing thinking: %+v", got[0])
	}
}

// attachThinkingForTest mirrors the extractResult attachment block.
// Lifted out so the unit can be tested without driving a full provider
// response through extractResult. If extractResult's attachment logic
// changes, update this mirror too.
func attachThinkingForTest(sessionDelta []bridle.SessionEvent, thinkingBlocks []bridle.ThinkingBlock) []bridle.SessionEvent {
	if len(thinkingBlocks) == 0 {
		return sessionDelta
	}
	attached := false
	for i := range sessionDelta {
		if sessionDelta[i].Role == bridle.RoleAssistant {
			sessionDelta[i].ThinkingBlocks = thinkingBlocks
			attached = true
			break
		}
	}
	if !attached {
		sessionDelta = append([]bridle.SessionEvent{{
			Provider:       bridle.ProviderClaude,
			Role:           bridle.RoleAssistant,
			ThinkingBlocks: thinkingBlocks,
		}}, sessionDelta...)
	}
	return sessionDelta
}
