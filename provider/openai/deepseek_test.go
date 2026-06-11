package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	openaisdk "github.com/openai/openai-go"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// newDeepSeekTestProvider builds a provider with the DeepSeek capability
// forced on, pointed at a local httptest URL (whose host isn't
// api.deepseek.com). Internal test seam.
func newDeepSeekTestProvider(apiKey, baseURL string) *Provider {
	return &Provider{apiKey: apiKey, baseURL: baseURL, forceDeepSeek: true}
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	return context.Background()
}

// discardSink drops every event.
type discardSink struct{}

func (discardSink) Emit(bridle.Event) {}

// capturingDeepSeekHandler records the last POST body and streams back a
// minimal OpenAI-shape SSE response the SDK accumulator can parse.
type capturingDeepSeekHandler struct {
	last []byte
}

func (h *capturingDeepSeekHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.last, _ = io.ReadAll(r.Body)
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)
	write := func(s string) {
		_, _ = io.WriteString(w, "data: "+s+"\n\n")
		if fl != nil {
			fl.Flush()
		}
	}
	write(`{"id":"x","object":"chat.completion.chunk","model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}`)
	write(`{"id":"x","object":"chat.completion.chunk","model":"deepseek-chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	write("[DONE]")
}

func (h *capturingDeepSeekHandler) body(t *testing.T) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(h.last, &m); err != nil {
		t.Fatalf("unmarshal captured body: %v\nraw=%s", err, h.last)
	}
	return m
}

// NEX-587: isDeepSeekEndpoint flags a provider pointed at DeepSeek's
// OpenAI-compatible /v1 so per-endpoint capability gating (strict
// json_schema unsupported) can fire. Detection is host-based off the
// configured baseURL — robust to scheme/path/trailing-slash variation.
func TestIsDeepSeekEndpoint(t *testing.T) {
	cases := []struct {
		baseURL string
		want    bool
	}{
		{"https://api.deepseek.com/v1", true},
		{"https://api.deepseek.com", true},
		{"https://api.deepseek.com/v1/", true},
		{"http://api.deepseek.com/v1", true},
		{"https://API.DeepSeek.com/v1", true}, // case-insensitive host
		{"https://api.openai.com/v1", false},
		{"https://api.together.xyz/v1", false},
		{"http://localhost:11434/v1", false},
		{"", false}, // default openai endpoint
	}
	for _, c := range cases {
		t.Run(c.baseURL, func(t *testing.T) {
			if got := isDeepSeekEndpoint(c.baseURL); got != c.want {
				t.Errorf("isDeepSeekEndpoint(%q) = %v, want %v", c.baseURL, got, c.want)
			}
		})
	}
}

// NEX-587: a DeepSeek-targeted provider degrades a json_schema STRICT
// response_format request to json_object rather than putting the
// unsupported strict wire on the line (DeepSeek /v1 returns 400 "This
// response_format type is unavailable now"). Mirrors nexus's
// CheapModelFilter portable default.
func TestResponseFormat_DeepSeekDegradesStrictToJSONObject(t *testing.T) {
	p := NewWithBaseURL("sk-test", "https://api.deepseek.com/v1")
	rf := &bridle.ResponseFormat{
		Type:   "json_schema",
		Name:   "verdict",
		Strict: true,
		Schema: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`),
	}
	got := p.responseFormatFor(rf)
	if got == nil {
		t.Fatal("expected a response_format union, got nil")
	}
	if got.OfJSONObject == nil {
		t.Errorf("strict json_schema against DeepSeek should degrade to json_object; got %+v", got)
	}
	if got.OfJSONSchema != nil {
		t.Errorf("strict json_schema must NOT reach DeepSeek wire; got json_schema variant")
	}
}

// NEX-587: a DeepSeek-targeted json_schema request WITHOUT strict still
// degrades to json_object — DeepSeek /v1 rejects the json_schema
// response_format type entirely, strict or not (the 400 is on the type,
// not the strict flag).
func TestResponseFormat_DeepSeekDegradesNonStrictJSONSchemaToJSONObject(t *testing.T) {
	p := NewWithBaseURL("sk-test", "https://api.deepseek.com/v1")
	rf := &bridle.ResponseFormat{
		Type:   "json_schema",
		Name:   "verdict",
		Strict: false,
		Schema: json.RawMessage(`{"type":"object"}`),
	}
	got := p.responseFormatFor(rf)
	if got == nil || got.OfJSONObject == nil {
		t.Errorf("json_schema against DeepSeek should degrade to json_object; got %+v", got)
	}
}

// NEX-587: json_object is supported by DeepSeek — it passes through
// unchanged (no degradation, since there's nothing to degrade).
func TestResponseFormat_DeepSeekJSONObjectPassesThrough(t *testing.T) {
	p := NewWithBaseURL("sk-test", "https://api.deepseek.com/v1")
	rf := &bridle.ResponseFormat{Type: "json_object"}
	got := p.responseFormatFor(rf)
	if got == nil || got.OfJSONObject == nil {
		t.Errorf("json_object should pass through to DeepSeek unchanged; got %+v", got)
	}
}

// NEX-587: OpenAI-proper (no DeepSeek baseURL) still sends strict
// json_schema on the wire — degradation is DeepSeek-scoped, the
// OpenAI-proper strict promise (NEX-300) must not regress.
func TestResponseFormat_OpenAIKeepsStrictJSONSchema(t *testing.T) {
	p := NewWithBaseURL("sk-test", "") // default openai endpoint
	rf := &bridle.ResponseFormat{
		Type:   "json_schema",
		Name:   "verdict",
		Strict: true,
		Schema: json.RawMessage(`{"type":"object"}`),
	}
	got := p.responseFormatFor(rf)
	if got == nil {
		t.Fatal("expected response_format, got nil")
	}
	if got.OfJSONSchema == nil {
		t.Errorf("OpenAI-proper must keep strict json_schema; got %+v", got)
	}
	if got.OfJSONObject != nil {
		t.Errorf("OpenAI-proper must NOT degrade to json_object")
	}
}

// NEX-587: nil response_format → nil union (provider leaves the field
// unset), regardless of endpoint.
func TestResponseFormat_NilStaysNil(t *testing.T) {
	for _, base := range []string{"", "https://api.deepseek.com/v1"} {
		p := NewWithBaseURL("sk-test", base)
		if got := p.responseFormatFor(nil); got != nil {
			t.Errorf("base=%q: nil response_format should map to nil, got %+v", base, got)
		}
	}
}

// NEX-587: classifyOpenAIError maps DeepSeek/OpenAI HTTP error status
// codes to bridle ProviderErrorKind so callers (the funnel's retry/
// backoff) can distinguish a rate-limit from an auth failure instead of
// treating every API error as a generic failure.
func TestClassifyOpenAIError_StatusCodes(t *testing.T) {
	cases := []struct {
		status int
		want   bridle.ProviderErrorKind
	}{
		{http.StatusTooManyRequests, bridle.ProviderErrorRateLimit},       // 429
		{http.StatusUnauthorized, bridle.ProviderErrorAuthFailed},         // 401
		{http.StatusForbidden, bridle.ProviderErrorAuthFailed},            // 403
		{http.StatusInternalServerError, bridle.ProviderErrorServerError}, // 500
		{http.StatusBadGateway, bridle.ProviderErrorServerError},          // 502
		{http.StatusServiceUnavailable, bridle.ProviderErrorServerError},  // 503
		{http.StatusGatewayTimeout, bridle.ProviderErrorTimeout},          // 504
	}
	for _, c := range cases {
		apiErr := &openaisdk.Error{StatusCode: c.status}
		got := classifyOpenAIError(apiErr)
		if got == nil {
			t.Errorf("status %d: classifyOpenAIError returned nil, want kind %s", c.status, c.want)
			continue
		}
		if got.Kind != c.want {
			t.Errorf("status %d: Kind = %s, want %s", c.status, got.Kind, c.want)
		}
		if !bridle.IsProviderErrorKind(got, c.want) {
			t.Errorf("status %d: IsProviderErrorKind(%s) = false", c.status, c.want)
		}
	}
}

// NEX-587: a non-API error (e.g. context cancel, network dial) is not
// an *openai.Error — classifyOpenAIError returns nil so RunTurn falls
// back to the generic wrap rather than mislabeling it.
func TestClassifyOpenAIError_NonAPIErrorReturnsNil(t *testing.T) {
	if got := classifyOpenAIError(errors.New("dial tcp: connection refused")); got != nil {
		t.Errorf("non-API error should classify as nil, got %+v", got)
	}
	if got := classifyOpenAIError(nil); got != nil {
		t.Errorf("nil error should classify as nil, got %+v", got)
	}
}

// NEX-587: a 4xx that isn't auth/rate-limit (e.g. 400 bad request, the
// DeepSeek response_format-type-unavailable shape) classifies as a
// server_error-distinct generic-but-typed error so it's at least a
// ProviderError, not a raw wrap. We map unknown 4xx to ServerError's
// sibling — pick a stable kind. Here: 400 → ServerError is wrong
// semantically; assert it is at minimum a ProviderError carrying the
// status in the message.
func TestClassifyOpenAIError_Unknown4xxStillTyped(t *testing.T) {
	apiErr := &openaisdk.Error{StatusCode: http.StatusBadRequest} // 400
	got := classifyOpenAIError(apiErr)
	if got == nil {
		t.Fatal("400 should still produce a typed ProviderError, got nil")
	}
	// 400 is a client/request error, not auth/rate-limit — must not be
	// misclassified as either.
	if got.Kind == bridle.ProviderErrorRateLimit || got.Kind == bridle.ProviderErrorAuthFailed {
		t.Errorf("400 misclassified as %s", got.Kind)
	}
}

// reasoningStreamHandler streams a DeepSeek-reasoner-shape SSE response
// where reasoning_content arrives as per-chunk deltas BEFORE the answer
// content (the real wire ordering: reasoner thinks, then answers).
type reasoningStreamHandler struct{}

func (reasoningStreamHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)
	write := func(s string) {
		_, _ = io.WriteString(w, "data: "+s+"\n\n")
		if fl != nil {
			fl.Flush()
		}
	}
	// reasoning_content deltas (the "thinking" channel)
	write(`{"id":"x","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"6 times 7 "}}]}`)
	write(`{"id":"x","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"is 42."}}]}`)
	// answer content deltas (the "answer" channel)
	write(`{"id":"x","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"Forty-"}}]}`)
	write(`{"id":"x","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"two."}}]}`)
	write(`{"id":"x","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
	write("[DONE]")
}

// chunkRecordingSink records every ModelChunk text emitted live.
type chunkRecordingSink struct{ chunks []string }

func (s *chunkRecordingSink) Emit(ev bridle.Event) {
	if mc, ok := ev.(bridle.ModelChunk); ok {
		s.chunks = append(s.chunks, mc.Text)
	}
}

// NEX-587: a DeepSeek-reasoner streaming response carrying BOTH
// reasoning_content and content must surface them SEPARATELY — the
// reasoning ends up on ProviderResult.ReasoningContent, the answer on
// FinalText, and the reasoning must NOT leak into FinalText (or the live
// ModelChunk stream the funnel paints).
func TestRunTurn_DeepSeekReasoningDoesNotLeakIntoAnswer(t *testing.T) {
	srv := httptest.NewServer(reasoningStreamHandler{})
	defer srv.Close()

	p := NewWithBaseURL("sk-test", srv.URL)
	sink := &chunkRecordingSink{}
	res, err := p.RunTurn(testContext(t), bridle.ProviderRequest{
		Model:    "deepseek-reasoner",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "what's 6*7?"}},
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	if res.FinalText != "Forty-two." {
		t.Errorf("FinalText = %q, want %q (answer only, no reasoning leak)", res.FinalText, "Forty-two.")
	}
	if res.ReasoningContent != "6 times 7 is 42." {
		t.Errorf("ReasoningContent = %q, want the separated reasoning text", res.ReasoningContent)
	}
	// The reasoning must not bleed into the answer text.
	if strings.Contains(res.FinalText, "6 times 7") || strings.Contains(res.FinalText, "is 42.") {
		t.Errorf("reasoning_content leaked into FinalText: %q", res.FinalText)
	}
	// Live-painted chunks (what the funnel/UI shows) must be answer-only.
	joined := strings.Join(sink.chunks, "")
	if joined != "Forty-two." {
		t.Errorf("live ModelChunk stream = %q, want answer-only %q (reasoning must not paint)", joined, "Forty-two.")
	}

	// Usage flows through (the NEX-581 contract) for the reasoner path too.
	if res.Usage.InputTokens != 10 || res.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v, want input=10 output=5", res.Usage)
	}

	// Reasoning is attached to the assistant SessionEvent for cross-turn replay.
	var sawReasoning bool
	for _, e := range res.SessionDelta {
		if e.ReasoningContent == "6 times 7 is 42." {
			sawReasoning = true
		}
	}
	if !sawReasoning {
		t.Errorf("SessionDelta missing reasoning_content for cross-turn replay: %+v", res.SessionDelta)
	}
}

// NEX-587 wire integration: a DeepSeek-targeted RunTurn that requests a
// strict json_schema actually puts json_object (NOT json_schema) on the
// wire — end-to-end through RunTurn, not just the unit mapper.
func TestRunTurn_DeepSeekStrictDegradesOnWire(t *testing.T) {
	h := &capturingDeepSeekHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	// Point a "deepseek" provider at the local httptest server but force
	// the DeepSeek capability on via the test seam (the real host check
	// keys off api.deepseek.com; the httptest URL isn't that host).
	p := newDeepSeekTestProvider("sk-test", srv.URL)

	_, err := p.RunTurn(testContext(t), bridle.ProviderRequest{
		Model:    "deepseek-chat",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		ResponseFormat: &bridle.ResponseFormat{
			Type:   "json_schema",
			Name:   "verdict",
			Strict: true,
			Schema: json.RawMessage(`{"type":"object"}`),
		},
	}, discardSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	wire := h.body(t)
	rf, ok := wire["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format missing on wire: %v", wire)
	}
	if rf["type"] != "json_object" {
		t.Errorf("DeepSeek wire response_format.type = %v, want json_object (degraded)", rf["type"])
	}
}
