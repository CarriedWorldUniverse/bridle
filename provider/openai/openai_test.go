package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/openai"
)

// capturingBodyHandler records the request body of every POST so
// tests can assert structural shape of what bridle puts on the wire.
type capturingBodyHandler struct {
	hits     atomic.Int32
	lastBody atomic.Value // []byte
}

func (h *capturingBodyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits.Add(1)
	body, _ := io.ReadAll(r.Body)
	h.lastBody.Store(body)
	streamMinimalOpenAIResponse(w)
}

// captureHandler records the path + auth of every request and streams
// back a minimal OpenAI-Chat-Completions SSE response so the SDK
// accumulator can parse it. Used to assert NewWithBaseURL routes
// requests through the override.
type captureHandler struct {
	hits     atomic.Int32
	lastPath atomic.Value // string
	lastAuth atomic.Value // string
}

func (h *captureHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits.Add(1)
	h.lastPath.Store(r.URL.Path)
	h.lastAuth.Store(r.Header.Get("Authorization"))
	streamMinimalOpenAIResponse(w)
}

// streamMinimalOpenAIResponse emits a tiny but well-formed SSE stream
// matching OpenAI's Chat Completions wire shape that openai-go's
// accumulator can assemble to a final ChatCompletion. Enough to
// round-trip a turn without touching the network.
func streamMinimalOpenAIResponse(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	write := func(data string) {
		_, _ = io.WriteString(w, "data: "+data+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}
	// First chunk: role + first content delta.
	write(`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}`)
	// Final chunk: finish_reason set, no more content.
	write(`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	write("[DONE]")
}

type nullSink struct{}

func (nullSink) Emit(_ bridle.Event) {}

// TestNewWithBaseURL_RoutesRequestsToOverride pins that the BaseURL
// option threads through to the SDK so requests hit the operator-
// specified endpoint instead of api.openai.com. Core of NEX-295:
// without it, DeepSeek's /v1, Together, Groq, Fireworks, Ollama,
// vLLM, LM Studio — all the OpenAI-compatible third-party services
// — can't be reached through this provider.
func TestNewWithBaseURL_RoutesRequestsToOverride(t *testing.T) {
	h := &captureHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn against override base URL: %v", err)
	}
	if h.hits.Load() == 0 {
		t.Fatal("override base URL never received a request — provider went somewhere else")
	}
	path, _ := h.lastPath.Load().(string)
	if !strings.Contains(path, "/chat/completions") {
		t.Errorf("expected /chat/completions on the override URL; got %q", path)
	}
	auth, _ := h.lastAuth.Load().(string)
	if !strings.Contains(auth, "sk-test-key") {
		t.Errorf("expected api key forwarded in Authorization; got %q", auth)
	}
}

// TestNew_NoBaseURL_PreservesDefaultBehaviour pins that the existing
// New(apiKey) constructor still works without a base URL — additive
// change must not regress callers that don't opt in.
func TestNew_NoBaseURL_PreservesDefaultBehaviour(t *testing.T) {
	p := openai.New("sk-test-key")
	caps := p.Capabilities()
	if !caps.SupportsCustomTools {
		t.Errorf("expected tools capability preserved on plain New; got %+v", caps)
	}
	if p.Name() != bridle.ProviderOpenAI {
		t.Errorf("expected ProviderOpenAI name; got %q", p.Name())
	}
}

// TestNewWithBaseURL_EmptyBaseURL_FallsBackToDefault pins the empty
// baseURL contract: NewWithBaseURL("k", "") behaves like New("k") —
// the SDK uses its built-in default endpoint.
func TestNewWithBaseURL_EmptyBaseURL_FallsBackToDefault(t *testing.T) {
	p := openai.NewWithBaseURL("sk-test-key", "")
	if p == nil {
		t.Fatal("NewWithBaseURL with empty baseURL must not return nil")
	}
	if p.Name() != bridle.ProviderOpenAI {
		t.Errorf("expected ProviderOpenAI name; got %q", p.Name())
	}
}

// TestNEX299P2_SamplingAndOutputFieldsThreadToWire pins NEX-299 Pass
// 2: TurnRequest's new fields (Temperature, TopP, Seed,
// MaxOutputTokens, StopSequences, ResponseFormat, ToolChoice) all
// reach the OpenAI wire format. Pre-Pass-2 these were silently
// dropped — ToolChoice in particular was a documented bridle field
// the openai provider just ignored.
func TestNEX299P2_SamplingAndOutputFieldsThreadToWire(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	temp := 0.0
	topP := 0.95
	seed := 42

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:           "test-model",
		Messages:        []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		Temperature:     &temp,
		TopP:            &topP,
		Seed:            &seed,
		MaxOutputTokens: 256,
		StopSequences:   []string{"END", "STOP"},
		ResponseFormat: &bridle.ResponseFormat{
			Type:        "json_schema",
			Name:        "verdict",
			Description: "judge verdict",
			Strict:      true,
			Schema:      json.RawMessage(`{"type":"object","properties":{"class":{"type":"string"}},"required":["class"],"additionalProperties":false}`),
		},
		ToolChoice: "any",
		Tools: []bridle.ToolDef{{
			Name:        "noop",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if _, present := wire["temperature"]; !present {
		t.Error("temperature key missing from wire — Temperature *float64 not threaded")
	}
	if v, _ := wire["temperature"].(float64); v != 0.0 {
		t.Errorf("temperature = %v, want 0.0", v)
	}
	if v, _ := wire["top_p"].(float64); v != 0.95 {
		t.Errorf("top_p = %v, want 0.95", v)
	}
	if v, _ := wire["seed"].(float64); v != 42 {
		t.Errorf("seed = %v, want 42", v)
	}
	if v, _ := wire["max_completion_tokens"].(float64); v != 256 {
		t.Errorf("max_completion_tokens = %v, want 256", v)
	}
	stops, _ := wire["stop"].([]any)
	if len(stops) != 2 || stops[0] != "END" {
		t.Errorf("stop = %v, want [END STOP]", stops)
	}
	// tool_choice: bridle "any" maps to OpenAI "required" (string variant)
	if tc := wire["tool_choice"]; tc != "required" {
		t.Errorf("tool_choice = %v, want \"required\" (bridle 'any' -> OpenAI 'required')", tc)
	}
	// response_format json_schema with strict
	rf, _ := wire["response_format"].(map[string]any)
	if rf == nil {
		t.Fatal("response_format missing")
	}
	if rf["type"] != "json_schema" {
		t.Errorf("response_format.type = %v, want json_schema", rf["type"])
	}
	js, _ := rf["json_schema"].(map[string]any)
	if js == nil {
		t.Fatal("response_format.json_schema missing")
	}
	if js["name"] != "verdict" {
		t.Errorf("response_format.json_schema.name = %v, want verdict", js["name"])
	}
	if js["strict"] != true {
		t.Errorf("response_format.json_schema.strict = %v, want true", js["strict"])
	}
	if _, ok := js["schema"].(map[string]any); !ok {
		t.Errorf("response_format.json_schema.schema should be an object, got %T", js["schema"])
	}
}

// TestNEX299P2_NamedToolChoiceUsesObjectVariant pins that asking for
// a specific tool by name produces the named-function tool_choice
// variant (not the string variant), matching OpenAI's wire spec.
func TestNEX299P2_NamedToolChoiceUsesObjectVariant(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:      "test-model",
		Messages:   []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		ToolChoice: "send_chat",
		Tools: []bridle.ToolDef{{
			Name:        "send_chat",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	_ = json.Unmarshal(body, &wire)
	tc, ok := wire["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice should be object variant for named-tool choice; got %T (%v)", wire["tool_choice"], wire["tool_choice"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_choice.type = %v, want function", tc["type"])
	}
	fn, _ := tc["function"].(map[string]any)
	if fn == nil || fn["name"] != "send_chat" {
		t.Errorf("tool_choice.function.name = %v, want send_chat", fn)
	}
}

// TestNEX299P2_NilOptionalFieldsOmittedFromWire pins back-compat:
// when caller doesn't set the new fields, none of them appear on
// the wire (nothing to break existing payload contracts).
func TestNEX299P2_NilOptionalFieldsOmittedFromWire(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		// All NEX-299 Pass 2 fields deliberately unset.
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	body, _ := h.lastBody.Load().([]byte)
	var wire map[string]any
	_ = json.Unmarshal(body, &wire)
	for _, k := range []string{"temperature", "top_p", "seed", "max_completion_tokens", "stop", "response_format", "tool_choice"} {
		if _, present := wire[k]; present {
			t.Errorf("%s should be absent when unset (back-compat); got value %v", k, wire[k])
		}
	}
}
