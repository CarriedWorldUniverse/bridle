package openai_test

import (
	"encoding/json"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// TestCrossTurn_DottedToolNameSurvivesTwoIndependentSanitizePasses is the
// property the pure-function design exists for: a dotted tool name must
// resolve to the IDENTICAL wire name whether it comes from the CURRENT
// turn's tool listing (toOpenAITools) or from REPLAYING a past tool call
// out of session history (toOpenAIMessages) — two independent call sites
// with no shared state between them. If they ever disagreed, the wire
// would carry two different spellings of the same tool across one
// request, which is either a 400 or a misrouted call depending on which
// half the API happens to validate.
func TestCrossTurn_DottedToolNameSurvivesTwoIndependentSanitizePasses(t *testing.T) {
	got := runWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Tools: []bridle.ToolDef{{
			Name:        "task.write",
			Description: "writes the task list",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "track this"},
			{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{
				ID:   "call_1",
				Name: "task.write", // the ORIGINAL name, as bridle.ToolInvocation always carries it
				Args: json.RawMessage(`{"tasks":[]}`),
			}}},
			{Role: "tool_result", ToolCallID: "call_1", Content: "ok"},
			{Role: "user", Content: "now what"},
		},
	})

	// The current-turn tools listing sanitized name.
	if len(got.Tools) != 1 {
		t.Fatalf("wire tools = %+v; want exactly 1", got.Tools)
	}
	fromListing := got.Tools[0]["function"].(map[string]interface{})["name"]

	// The replayed history's sanitized name.
	asst := got.Messages[1]
	calls := asst["tool_calls"].([]interface{})
	fromHistory := calls[0].(map[string]interface{})["function"].(map[string]interface{})["name"]

	if fromListing != "task_write" {
		t.Errorf("tools listing wire name = %v; want task_write", fromListing)
	}
	if fromHistory != "task_write" {
		t.Errorf("history replay wire name = %v; want task_write", fromHistory)
	}
	if fromListing != fromHistory {
		t.Fatalf("the two independent sanitize call sites DISAGREED: listing=%v history=%v", fromListing, fromHistory)
	}
}

func TestCrossTurn_ToolResultForADottedNameStillMatchesByCallID(t *testing.T) {
	// tool_result matches by tool_call_id, not by name — confirm the
	// sanitization doesn't disturb that at all.
	got := runWithCapture(t, bridle.ProviderRequest{
		Model: "test-model",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "go"},
			{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{
				ID: "call_xyz", Name: "memory.write", Args: json.RawMessage(`{}`),
			}}},
			{Role: "tool_result", ToolCallID: "call_xyz", Content: "stored"},
		},
	})
	tool := got.Messages[2]
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_xyz" {
		t.Fatalf("tool_result message malformed: %+v", tool)
	}
}
