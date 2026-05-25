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
