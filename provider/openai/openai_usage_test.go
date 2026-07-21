package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/openai"
)

// TestUsage_StreamingRequestsIncludeUsage — the usage contract
// (NEX-581/589) requires the streaming request to carry
// stream_options.include_usage so the final chunk reports tokens.
// Without it, the openai-compat shim (ollama /v1) and vLLM report zero.
// Assert the flag lands on the wire body.
func TestUsage_StreamingRequestsIncludeUsage(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	if _, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, nullSink{}); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	raw, _ := h.lastBody.Load().([]byte)
	if len(raw) == 0 {
		t.Fatal("no request body captured")
	}
	var got struct {
		Stream        bool `json:"stream"`
		StreamOptions struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode body: %v\nbody=%s", err, raw)
	}
	if !got.StreamOptions.IncludeUsage {
		t.Errorf("stream_options.include_usage not set on the wire; body=%s", raw)
	}
}

// TestUsage_ExtractedFromResponse — a streaming response that carries a
// usage block (the server honored include_usage) yields real,
// non-estimated counts on the ProviderResult.
func TestUsage_ExtractedFromResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		streamWithUsage(w, 17, 5)
	}))
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	res, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if res.Usage.InputTokens != 17 || res.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want Input=17 Output=5", res.Usage)
	}
	if res.Usage.Estimated {
		t.Error("Estimated = true, want false when the engine reported real usage")
	}
}

// TestUsage_NoUsageBlockReturnsZero — a compat server that ignores
// include_usage and omits the usage block leaves the provider-level
// Usage zero (and not flagged). The estimated floor is applied one
// layer up in run.go's normalizeUsage, not in the provider — the
// provider's job is just to extract what the engine gave (here:
// nothing).
func TestUsage_NoUsageBlockReturnsZero(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		streamNoUsage(w)
	}))
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	res, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if res.Usage.InputTokens != 0 || res.Usage.OutputTokens != 0 {
		t.Errorf("usage = %+v, want zero (estimated floor is applied in run.go, not the provider)", res.Usage)
	}
}

func sseWrite(w http.ResponseWriter, data string) {
	_, _ = io.WriteString(w, "data: "+data+"\n\n")
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// streamWithUsage emits a well-formed SSE stream whose final chunk
// carries a usage block (server honored include_usage).
func streamWithUsage(w http.ResponseWriter, prompt, completion int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	sseWrite(w, `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}`)
	final := `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":` +
		itoa(prompt) + `,"completion_tokens":` + itoa(completion) + `,"total_tokens":` + itoa(prompt+completion) + `}}`
	sseWrite(w, final)
	sseWrite(w, "[DONE]")
}

// streamNoUsage emits a well-formed SSE stream with NO usage block —
// the compat-server-ignores-the-flag case.
func streamNoUsage(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	sseWrite(w, `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}`)
	sseWrite(w, `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	sseWrite(w, "[DONE]")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestUsage_OpenRouterCostExtraField — OpenRouter (and LiteLLM forwarding
// it) reports the exact upstream charge as a non-standard `cost` field on
// the usage block. The provider must lower it into Usage.CostUSD — and the
// stream accumulator must not drop the extra field on the way through.
func TestUsage_OpenRouterCostExtraField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		sseWrite(w, `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}`)
		sseWrite(w, `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1310,"completion_tokens":16,"total_tokens":1326,"cost":0.00417,"prompt_tokens_details":{"cached_tokens":1280}}}`)
		sseWrite(w, "[DONE]")
	}))
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	res, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if res.Usage.CostUSD != 0.00417 {
		t.Errorf("CostUSD = %v, want 0.00417 (the OpenRouter-reported charge)", res.Usage.CostUSD)
	}
	if res.Usage.CacheReadInputTokens != 1280 {
		t.Errorf("CacheReadInputTokens = %d, want 1280", res.Usage.CacheReadInputTokens)
	}
	// bridle.Usage's contract: InputTokens = UNCACHED prompt tokens. The
	// OpenAI-shape prompt_tokens total (1310) INCLUDES the 1280 cached —
	// lowering must subtract, or claude-lane (natively disjoint) and
	// openai-lane usage mean different things and every consumer's cache%%
	// and billing math is wrong for one of them.
	if res.Usage.InputTokens != 30 {
		t.Errorf("InputTokens = %d, want 30 (1310 total - 1280 cached; uncached-only contract)", res.Usage.InputTokens)
	}
}

// TestUsage_NoCostFieldZero — standard OpenAI backends have no `cost`
// extra field; CostUSD stays zero (host may price-table it instead).
func TestUsage_NoCostFieldZero(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		streamWithUsage(w, 17, 5)
	}))
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	res, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if res.Usage.CostUSD != 0 {
		t.Errorf("CostUSD = %v, want 0 when the backend reports no cost", res.Usage.CostUSD)
	}
}
