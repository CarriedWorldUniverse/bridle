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
