package claude_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/claude"
)

// TestThinkingBudget_EnabledAndMaxTokensBumped pins the request-side
// extended-thinking budget: ThinkingBudgetTokens >= 1024 must set
// "thinking":{"type":"enabled","budget_tokens":N} on the wire, AND
// max_tokens must come out strictly greater than the budget (Anthropic
// 400s otherwise). Here MaxOutputTokens is left unset (falls back to
// the historical 4096 default), which is <= the 5000 budget requested,
// so the provider must bump max_tokens itself.
func TestThinkingBudget_EnabledAndMaxTokensBumped(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:                "test-model",
		Messages:             []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		ThinkingBudgetTokens: 5000,
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode wire body: %v\nbody=%s", err, string(body))
	}

	thinking, _ := wire["thinking"].(map[string]any)
	if thinking == nil {
		t.Fatalf("thinking missing from wire\nbody=%s", string(body))
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want enabled", thinking["type"])
	}
	if v, _ := thinking["budget_tokens"].(float64); v != 5000 {
		t.Errorf("thinking.budget_tokens = %v, want 5000", v)
	}

	maxTokens, _ := wire["max_tokens"].(float64)
	if maxTokens <= 5000 {
		t.Errorf("max_tokens = %v, want > 5000 (budget_tokens must be < max_tokens)", maxTokens)
	}
}

// TestThinkingBudget_ZeroLeavesThinkingUnset pins the no-op default:
// ThinkingBudgetTokens==0 (the zero value every existing caller has)
// must not add a "thinking" key to the wire at all.
func TestThinkingBudget_ZeroLeavesThinkingUnset(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		// ThinkingBudgetTokens deliberately unset (== 0)
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	_ = json.Unmarshal(body, &wire)
	if _, present := wire["thinking"]; present {
		t.Errorf("thinking present on wire when ThinkingBudgetTokens==0\nbody=%s", string(body))
	}
}

// TestThinkingBudget_SubFloorRoundsUpTo1024 pins the caller-friendly
// clamp: a caller requesting "some" thinking with 0 < budget < 1024
// (below Anthropic's floor) gets rounded UP to 1024 rather than
// rejected or silently dropped.
func TestThinkingBudget_SubFloorRoundsUpTo1024(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:                "test-model",
		Messages:             []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		ThinkingBudgetTokens: 100,
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode wire body: %v\nbody=%s", err, string(body))
	}
	thinking, _ := wire["thinking"].(map[string]any)
	if thinking == nil {
		t.Fatalf("thinking missing from wire\nbody=%s", string(body))
	}
	if v, _ := thinking["budget_tokens"].(float64); v != 1024 {
		t.Errorf("thinking.budget_tokens = %v, want 1024 (rounded up from 100)", v)
	}
	maxTokens, _ := wire["max_tokens"].(float64)
	if maxTokens <= 1024 {
		t.Errorf("max_tokens = %v, want > 1024", maxTokens)
	}
}

// TestThinkingBudget_UnderExistingMaxTokensNotBumped pins that when the
// caller's MaxOutputTokens is already comfortably above the budget, the
// provider leaves max_tokens exactly as the caller set it (no
// surprise inflation).
func TestThinkingBudget_UnderExistingMaxTokensNotBumped(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:                "test-model",
		Messages:             []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		ThinkingBudgetTokens: 1024,
		MaxOutputTokens:      8192,
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	_ = json.Unmarshal(body, &wire)
	if v, _ := wire["max_tokens"].(float64); v != 8192 {
		t.Errorf("max_tokens = %v, want unchanged 8192", v)
	}
}
