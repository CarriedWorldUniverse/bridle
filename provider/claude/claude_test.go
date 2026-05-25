package claude_test

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
	"github.com/CarriedWorldUniverse/bridle/provider/claude"
)

// captureHandler records the path + auth header of every request and
// streams back a minimal Anthropic-shape SSE response so the SDK's
// stream parser is happy. Used to assert NewWithBaseURL routes
// requests through the override.
type captureHandler struct {
	hits      atomic.Int32
	lastPath  atomic.Value // string
	lastAuth  atomic.Value // string
	respond   func(w http.ResponseWriter, r *http.Request)
}

func (h *captureHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits.Add(1)
	h.lastPath.Store(r.URL.Path)
	h.lastAuth.Store(r.Header.Get("X-Api-Key"))
	if h.respond != nil {
		h.respond(w, r)
		return
	}
	streamMinimalAnthropicResponse(w)
}

// streamMinimalAnthropicResponse emits a tiny but well-formed SSE
// stream that the anthropic-sdk-go accumulator can parse to a final
// Message. Enough to round-trip a turn without external network.
func streamMinimalAnthropicResponse(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	write := func(event, data string) {
		_, _ = io.WriteString(w, "event: "+event+"\n")
		_, _ = io.WriteString(w, "data: "+data+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}
	write("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"test-model","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`)
	write("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
	write("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`)
	write("content_block_stop", `{"type":"content_block_stop","index":0}`)
	write("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`)
	write("message_stop", `{"type":"message_stop"}`)
}

type nullSink struct{}

func (nullSink) Emit(_ bridle.Event) {}

// TestNewWithBaseURL_RoutesRequestsToOverride pins that the BaseURL
// option threads through to the SDK so requests hit the operator-
// specified endpoint instead of api.anthropic.com. This is the core
// of NEX-295: without it, third-party Anthropic-compat providers
// (DeepSeek's /anthropic endpoint, Anthropic-shape gateways, etc.)
// can't be reached.
func TestNewWithBaseURL_RoutesRequestsToOverride(t *testing.T) {
	h := &captureHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
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
	if !strings.Contains(path, "/messages") {
		t.Errorf("expected /messages on the override URL; got %q", path)
	}
	auth, _ := h.lastAuth.Load().(string)
	if auth != "sk-test-key" {
		t.Errorf("expected api key header forwarded; got %q", auth)
	}
}

// TestNew_NoBaseURL_PreservesDefaultBehaviour pins that the existing
// New(apiKey) constructor still works without a base URL — additive
// change must not regress callers that don't opt in. Asserts via the
// Capabilities surface (doesn't make a real network call).
func TestNew_NoBaseURL_PreservesDefaultBehaviour(t *testing.T) {
	p := claude.New("sk-test-key")
	caps := p.Capabilities()
	if !caps.SupportsCustomTools {
		t.Errorf("expected tools capability preserved on plain New; got %+v", caps)
	}
	if p.Name() != bridle.ProviderClaude {
		t.Errorf("expected ProviderClaude name; got %q", p.Name())
	}
}

// TestNewWithBaseURL_EmptyBaseURL_FallsBackToDefault pins the empty
// baseURL contract: NewWithBaseURL("k", "") behaves like New("k") —
// the SDK uses its built-in default endpoint. Confirms by asserting
// the provider doesn't fail construction; a real network call would
// hit api.anthropic.com so we don't actually fire one.
func TestNewWithBaseURL_EmptyBaseURL_FallsBackToDefault(t *testing.T) {
	p := claude.NewWithBaseURL("sk-test-key", "")
	if p == nil {
		t.Fatal("NewWithBaseURL with empty baseURL must not return nil")
	}
	if p.Name() != bridle.ProviderClaude {
		t.Errorf("expected ProviderClaude name; got %q", p.Name())
	}
}

// silence unused-import lint when respond callback isn't used.
var _ = json.Marshal

// capturingBodyHandler records the raw request body of every POST so
// tests can assert structural shape of what bridle puts on the wire.
type capturingBodyHandler struct {
	hits     atomic.Int32
	lastBody atomic.Value // []byte
}

func (h *capturingBodyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits.Add(1)
	body, _ := io.ReadAll(r.Body)
	h.lastBody.Store(body)
	streamMinimalAnthropicResponse(w)
}

// TestToolSchemaWirePayloadIsCorrectlyShaped pins NEX-299 Pass 1:
// the tool's JSON Schema must hit the wire as a structurally correct
// Anthropic ToolInputSchemaParam — with properties + required at the
// top level, NOT the entire schema object nested inside Properties.
//
// Pre-fix the wire payload looked like:
//
//	"input_schema": {
//	  "type": "object",
//	  "properties": {
//	    "type": "object",
//	    "properties": {"text": {"type": "string"}},
//	    "required": ["text"]
//	  }
//	}
//
// Real Anthropic accepted this with lenient parsing; DeepSeek's
// strict /anthropic validator rejected the malformed `required`
// array showing up as a property entry. This test asserts the
// post-fix shape: properties.text exists at the right depth, and
// the inner properties does NOT contain "type"/"properties"/
// "required" keys (which would indicate the bug regressed).
func TestToolSchemaWirePayloadIsCorrectlyShaped(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		Tools: []bridle.ToolDef{{
			Name:        "echo",
			Description: "Echo back the input.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`),
		}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	bodyBytes, _ := h.lastBody.Load().([]byte)
	if len(bodyBytes) == 0 {
		t.Fatal("no request body captured")
	}
	var wire struct {
		Tools []struct {
			Name        string `json:"name"`
			InputSchema struct {
				Type       string         `json:"type"`
				Properties map[string]any `json:"properties"`
				Required   []string       `json:"required"`
			} `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(bodyBytes, &wire); err != nil {
		t.Fatalf("decode wire body: %v\nbody=%s", err, string(bodyBytes))
	}
	if len(wire.Tools) != 1 {
		t.Fatalf("expected 1 tool on wire, got %d\nbody=%s", len(wire.Tools), string(bodyBytes))
	}
	tool := wire.Tools[0]
	if tool.Name != "echo" {
		t.Errorf("tool name = %q, want echo", tool.Name)
	}
	// properties.text must exist at depth 1 — NOT nested under
	// properties.properties.text (the bug shape).
	textProp, ok := tool.InputSchema.Properties["text"]
	if !ok {
		t.Errorf("input_schema.properties.text missing — schema likely nested incorrectly\nbody=%s", string(bodyBytes))
	}
	if _, isMap := textProp.(map[string]any); !isMap {
		t.Errorf("input_schema.properties.text should be an object, got %T", textProp)
	}
	// required must be at the top level of input_schema, NOT inside properties
	if len(tool.InputSchema.Required) != 1 || tool.InputSchema.Required[0] != "text" {
		t.Errorf("input_schema.required = %v, want [text]", tool.InputSchema.Required)
	}
	// Regression guard: properties must NOT contain "type" / "properties" / "required"
	// keys — those would mean the full schema object got dumped in.
	for _, badKey := range []string{"type", "properties", "required"} {
		if _, exists := tool.InputSchema.Properties[badKey]; exists {
			t.Errorf("input_schema.properties.%s exists — the pre-fix bug has regressed\nbody=%s", badKey, string(bodyBytes))
		}
	}
}

// TestToolSchemaWithNoRequiredField pins that schemas without a
// "required" array still serialise cleanly. Important because the
// fix's `parsed.Required` will be nil; we want that to omit the
// field rather than send an empty array (Anthropic accepts both,
// but cleaner wire output is friendlier to strict validators).
func TestToolSchemaWithNoRequiredField(t *testing.T) {
	h := &capturingBodyHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := claude.NewWithBaseURL("sk-test-key", srv.URL)
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		Tools: []bridle.ToolDef{{
			Name:        "ping",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	bodyBytes, _ := h.lastBody.Load().([]byte)
	var wire struct {
		Tools []struct {
			InputSchema map[string]any `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(bodyBytes, &wire); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(wire.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(wire.Tools))
	}
	// Schema must have properties (even if empty map); required field
	// should be absent / null when no required keys in source.
	if _, ok := wire.Tools[0].InputSchema["properties"]; !ok {
		t.Errorf("input_schema.properties absent\nbody=%s", string(bodyBytes))
	}
}
