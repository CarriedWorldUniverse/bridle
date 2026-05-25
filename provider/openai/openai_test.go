package openai_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/openai"
)

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
