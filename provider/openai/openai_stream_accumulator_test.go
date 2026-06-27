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

// Streaming tool_call accumulation: OpenAI's Chat Completions
// streaming wire splits a single tool_call across many chunks. The
// `id`/`name` arrive on the first delta; subsequent deltas carry
// only `function.arguments` fragments. All deltas for the same
// tool_call share the same `index`.
//
// Example wire (one tool_call, arguments split across 4 deltas):
//
//	{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"send_chat","arguments":""}}]}}
//	{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"con"}}]}}
//	{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"tent"}}]}}
//	{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\"hello\"}"}}]}}
//
// bridle relies on openai-go's ChatCompletionAccumulator to merge
// these into one tool_call. If the accumulator (or our consumption
// of it) produces N ToolInvocations instead of 1, we get the plumb
// pathology: one logical send_chat fans out into N per-word calls.
//
// These tests pin the merge behaviour deterministically — they don't
// touch a live API.

// streamSplitToolCall emits an SSE stream where a single tool_call's
// arguments are split across argSplits chunks. Mirrors the openai
// wire format (per-delta tool_calls[] with matching index).
func streamSplitToolCall(w http.ResponseWriter, id, name string, argSplits []string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	write := func(data string) {
		_, _ = io.WriteString(w, "data: "+data+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}
	// Opening chunk: id + name + empty arguments.
	write(`{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"` + id + `","type":"function","function":{"name":"` + name + `","arguments":""}}]},"finish_reason":null}]}`)
	// Argument fragment chunks: index-only, no id/name.
	for _, frag := range argSplits {
		// JSON-escape the fragment so embedded quotes/backslashes survive.
		escaped, _ := json.Marshal(frag)
		write(`{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":` + string(escaped) + `}}]},"finish_reason":null}]}`)
	}
	// Finish chunk: finish_reason=tool_calls.
	write(`{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	write("[DONE]")
}

// splitToolCallHandler serves one fake stream per request — caller
// configures id/name/argSplits via fields, then the handler emits
// that shape exactly once.
type splitToolCallHandler struct {
	id        string
	name      string
	argSplits []string
}

func (h *splitToolCallHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	streamSplitToolCall(w, h.id, h.name, h.argSplits)
}

// TestStreamAccumulator_SingleToolCallSplitArguments — the failure
// mode plumb hits if the accumulator doesn't merge: a model that
// emits one send_chat with content="Done. Asked the operator to
// clarify..." gets streamed as one tool_call whose arguments JSON
// arrives split across N deltas. The accumulator must merge to ONE
// ToolInvocation. If it produces N, plumb fans the reply into N
// per-token send_chats — exactly the chunking the operator is seeing.
func TestStreamAccumulator_SingleToolCallSplitArguments(t *testing.T) {
	// Arguments JSON `{"content":"hello world"}` split into 6 frags.
	splits := []string{`{"co`, `nten`, `t":"`, `hello`, ` world`, `"}`}
	h := &splitToolCallHandler{id: "call_1", name: "send_chat", argSplits: splits}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	result, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "say hi"}},
		Tools: []bridle.ToolDef{{
			Name:        "send_chat",
			Description: "post a message",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}`),
		}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	if got := len(result.ToolCalls); got != 1 {
		t.Fatalf("got %d ToolCalls, want 1 (accumulator should merge split-argument deltas into one call); calls=%+v", got, result.ToolCalls)
	}
	tc := result.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Errorf("ToolCall ID = %q, want call_1", tc.ID)
	}
	if tc.Name != "send_chat" {
		t.Errorf("ToolCall Name = %q, want send_chat", tc.Name)
	}
	// Args must round-trip to the full original payload after merge.
	var args struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(tc.Args, &args); err != nil {
		t.Fatalf("merged Args don't parse: %v\nargs=%s", err, string(tc.Args))
	}
	if args.Content != "hello world" {
		t.Errorf("merged content = %q, want %q", args.Content, "hello world")
	}
}

// TestStreamAccumulator_TwoParallelToolCalls — model emits two
// parallel tool_calls in one turn (the "parallel_tool_calls=true"
// path, OpenAI's SDK default). Each has its own `index`. Streaming
// interleaves their argument deltas. The accumulator must produce
// exactly 2 ToolInvocations with correct per-call argument
// reassembly — not 1 (drop) or N (over-split).
func TestStreamAccumulator_TwoParallelToolCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		write := func(data string) {
			_, _ = io.WriteString(w, "data: "+data+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
		// Opening: both tool_calls declared with id+name, empty args.
		write(`{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"send_chat","arguments":""}},{"index":1,"id":"call_b","type":"function","function":{"name":"send_chat","arguments":""}}]},"finish_reason":null}]}`)
		// Interleaved argument fragments for both calls.
		write(`{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"content\":\"hi\"}"}}]},"finish_reason":null}]}`)
		write(`{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"content\":\"there\"}"}}]},"finish_reason":null}]}`)
		write(`{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		write("[DONE]")
	}))
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	result, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "say hi twice"}},
		Tools: []bridle.ToolDef{{
			Name:        "send_chat",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"content":{"type":"string"}}}`),
		}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if got := len(result.ToolCalls); got != 2 {
		t.Fatalf("got %d ToolCalls, want 2; calls=%+v", got, result.ToolCalls)
	}
	if result.ToolCalls[0].ID != "call_a" || result.ToolCalls[1].ID != "call_b" {
		t.Errorf("call IDs = [%q,%q], want [call_a,call_b]", result.ToolCalls[0].ID, result.ToolCalls[1].ID)
	}
	// Per-call argument reassembly: each call must have its own full
	// arguments payload, not a mashup of the other's.
	for i, want := range []string{"hi", "there"} {
		var args struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal(result.ToolCalls[i].Args, &args); err != nil {
			t.Fatalf("call[%d] args don't parse: %v\nargs=%s", i, err, string(result.ToolCalls[i].Args))
		}
		if args.Content != want {
			t.Errorf("call[%d].content = %q, want %q", i, args.Content, want)
		}
	}
}

// TestStreamAccumulator_FineGrainedSplit_ManyDeltas — extreme case:
// one tool_call's arguments split into ONE BYTE per delta. Mimics a
// pathological DeepSeek stream where the model emits character-level
// deltas. If the accumulator can handle this, regular 4-8 byte
// splits are fine too. If this test produces >1 ToolInvocation,
// that's the smoking gun for plumb's per-word fanout.
func TestStreamAccumulator_FineGrainedSplit_ManyDeltas(t *testing.T) {
	args := `{"content":"hello world"}`
	splits := make([]string, len(args))
	for i := 0; i < len(args); i++ {
		splits[i] = string(args[i])
	}
	h := &splitToolCallHandler{id: "call_byte", name: "send_chat", argSplits: splits}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := openai.NewWithBaseURL("sk-test-key", srv.URL)
	result, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "test-model",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "say hi"}},
		Tools: []bridle.ToolDef{{
			Name:        "send_chat",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"content":{"type":"string"}}}`),
		}},
	}, nullSink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if got := len(result.ToolCalls); got != 1 {
		t.Fatalf("byte-split produced %d ToolCalls, want 1 — this is the plumb chunking pathology", got)
	}
	if string(result.ToolCalls[0].Args) != args {
		t.Errorf("reassembled args = %q, want %q", string(result.ToolCalls[0].Args), args)
	}
}
