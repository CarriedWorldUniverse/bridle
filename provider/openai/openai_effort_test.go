package openai_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/openai"
)

// TestEffort_ThreadsToWire_High pins that a bridle Effort value reaches
// the OpenAI wire as reasoning_effort verbatim for a tier OpenAI has a
// direct name for.
func TestEffort_ThreadsToWire_High(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		Effort:   "high",
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if v, present := wire["reasoning_effort"]; !present || v != "high" {
		t.Errorf("reasoning_effort = %v (present=%v), want \"high\"", v, present)
	}
}

// TestEffort_EmptyOmittedFromWire pins that an empty Effort produces no
// reasoning_effort key on the wire (provider-default posture, same as
// every other NEX-299 Pass 2 optional field).
func TestEffort_EmptyOmittedFromWire(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		// Effort deliberately left empty.
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	_ = json.Unmarshal(body, &wire)
	if v, present := wire["reasoning_effort"]; present {
		t.Errorf("reasoning_effort should be absent when Effort is empty; got %v", v)
	}
}

// TestEffort_XhighAndMaxMapToHigh pins the agora-ladder-to-OpenAI-tier
// clamp: OpenAI's Chat Completions API caps at "high" — there is no
// higher tier to ask for — so both xhigh and max (bridle's two tiers
// above high) map onto OpenAI's ceiling rather than being dropped.
func TestEffort_XhighAndMaxMapToHigh(t *testing.T) {
	for _, effort := range []string{"xhigh", "max"} {
		t.Run(effort, func(t *testing.T) {
			h := &capturingBodyHandler{}
			srv := httptest.NewServer(h)
			defer srv.Close()

			p := openai.NewWithBaseURL("sk-test-key", srv.URL)
			_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
				Model:    "test-model",
				Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
				Effort:   effort,
			}, nullSink{})
			if err != nil {
				t.Fatalf("RunTurn: %v", err)
			}

			body, _ := h.lastBody.Load().([]byte)
			var wire map[string]any
			_ = json.Unmarshal(body, &wire)
			if v, present := wire["reasoning_effort"]; !present || v != "high" {
				t.Errorf("Effort=%q: reasoning_effort = %v (present=%v), want \"high\"", effort, v, present)
			}
		})
	}
}
