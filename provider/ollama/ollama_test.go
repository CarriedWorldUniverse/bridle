package ollama_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/provider/ollama"
)

// capturingChatHandler records the request body of every POST to the
// fake /api/chat endpoint and streams back a minimal NDJSON response
// (the ollama client reads application/x-ndjson, one JSON object per
// line) so client.Chat round-trips without a real server.
type capturingChatHandler struct {
	hits     atomic.Int32
	lastBody atomic.Value // []byte
}

func (h *capturingChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits.Add(1)
	body, _ := io.ReadAll(r.Body)
	h.lastBody.Store(body)

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, `{"model":"test-model","created_at":"2026-06-10T00:00:00Z","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`+"\n")
}

func (h *capturingChatHandler) body(t *testing.T) map[string]any {
	t.Helper()
	raw, _ := h.lastBody.Load().([]byte)
	if len(raw) == 0 {
		t.Fatal("fake ollama endpoint never received a request body")
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("request body is not valid JSON: %v\nbody: %s", err, raw)
	}
	return m
}

type nullSink struct{}

func (nullSink) Emit(_ bridle.Event) {}

func runTurn(t *testing.T, p *ollama.Provider) {
	t.Helper()
	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn against fake ollama endpoint: %v", err)
	}
}

// TestRunTurn_OptionsPassthrough pins the wire shape when the operator
// sets KeepAlive, NumCtx, and model options: keep_alive serializes via
// api.Duration.MarshalJSON as the Go duration string ("45m0s"), and
// options carries num_ctx alongside the passthrough options.
func TestRunTurn_OptionsPassthrough(t *testing.T) {
	h := &capturingChatHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := ollama.NewWithURL(srv.URL)
	p.KeepAlive = 45 * time.Minute
	p.NumCtx = 8192
	p.Options = map[string]any{"temperature": 0.2}

	runTurn(t, p)

	body := h.body(t)
	if got, want := body["keep_alive"], "45m0s"; got != want {
		t.Errorf("keep_alive = %#v, want %q", got, want)
	}
	opts, ok := body["options"].(map[string]any)
	if !ok {
		t.Fatalf("options is %#v, want a JSON object", body["options"])
	}
	if got, want := opts["num_ctx"], float64(8192); got != want {
		t.Errorf("options.num_ctx = %#v, want %v", got, want)
	}
	if got, want := opts["temperature"], 0.2; got != want {
		t.Errorf("options.temperature = %#v, want %v", got, want)
	}
}

// TestRunTurn_ZeroValueDefaults pins the zero-value contract: an
// unconfigured Provider sends the 30m keep_alive default (keel is
// always-on; ollama's server-side ~5m default would unload gemma
// between quiet periods) and omits num_ctx so the model default holds.
func TestRunTurn_ZeroValueDefaults(t *testing.T) {
	h := &capturingChatHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := ollama.NewWithURL(srv.URL)
	runTurn(t, p)

	body := h.body(t)
	if got, want := body["keep_alive"], "30m0s"; got != want {
		t.Errorf("keep_alive = %#v, want %q", got, want)
	}
	opts, ok := body["options"].(map[string]any)
	if !ok {
		t.Fatalf("options is %#v, want a JSON object", body["options"])
	}
	if _, present := opts["num_ctx"]; present {
		t.Errorf("options.num_ctx = %#v, want key absent (model default)", opts["num_ctx"])
	}
}

// TestRunTurn_DoesNotMutateProviderOptions pins that a turn works on a
// copy of p.Options: num_ctx is injected into the request, never into
// the Provider's own map, so repeated turns and shared Providers stay
// clean.
func TestRunTurn_DoesNotMutateProviderOptions(t *testing.T) {
	h := &capturingChatHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := ollama.NewWithURL(srv.URL)
	p.NumCtx = 8192
	p.Options = map[string]any{"temperature": 0.2}
	want := map[string]any{"temperature": 0.2}

	runTurn(t, p)
	runTurn(t, p)

	if !reflect.DeepEqual(p.Options, want) {
		t.Errorf("p.Options mutated by RunTurn: got %#v, want %#v", p.Options, want)
	}
}

// TestRunTurn_NumCtxWinsConflict pins the documented precedence: an
// explicit NumCtx overrides a conflicting num_ctx in Options.
func TestRunTurn_NumCtxWinsConflict(t *testing.T) {
	h := &capturingChatHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := ollama.NewWithURL(srv.URL)
	p.NumCtx = 8192
	p.Options = map[string]any{"num_ctx": 4096}

	runTurn(t, p)

	body := h.body(t)
	opts, ok := body["options"].(map[string]any)
	if !ok {
		t.Fatalf("options is %#v, want a JSON object", body["options"])
	}
	if got, want := opts["num_ctx"], float64(8192); got != want {
		t.Errorf("options.num_ctx = %#v, want %v (NumCtx must win over Options)", got, want)
	}
}
