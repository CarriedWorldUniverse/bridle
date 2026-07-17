package bridle_test

import (
	"context"
	"encoding/json"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
	"github.com/CarriedWorldUniverse/bridle/internal/normalize"
)

// drainStream collects every StreamEvent off ch until it closes.
func drainStream(ch <-chan bridle.StreamEvent) []bridle.StreamEvent {
	var out []bridle.StreamEvent
	for ev := range ch {
		out = append(out, ev)
	}
	return out
}

// TestStream_DirectAPI_EventSequence drives the RunStep (direct-api)
// path via a fake.Provider and asserts the normalized StreamEvent
// sequence: text_delta -> tool_call x N (assembled + IN ORDER) -> usage
// -> done{stop_reason}.
func TestStream_DirectAPI_EventSequence(t *testing.T) {
	toolCalls := []bridle.ToolInvocation{
		{ID: "call_1", Name: "search", Args: json.RawMessage(`{"q":"first"}`)},
		{ID: "call_2", Name: "fetch", Args: json.RawMessage(`{"url":"second"}`)},
	}
	provider := fake.NewProvider(fake.Step{
		Text:      "thinking about it",
		ToolCalls: toolCalls,
		Usage:     bridle.Usage{InputTokens: 10, OutputTokens: 5, CacheReadInputTokens: 2},
	})
	h := bridle.NewHarness(provider)

	r := bridle.NewRegistry()
	if err := r.Bind("claude-api", h); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	handle, err := r.Resolve("claude-api/claude-sonnet-5", "")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}

	ch, err := r.Stream(context.Background(), handle, bridle.Request{
		Messages: []bridle.RoleMessage{
			{Role: bridle.MessageRoleSystem, Content: "you are terse"},
			{Role: bridle.MessageRoleUser, Content: "search for x"},
		},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainStream(ch)

	if len(events) != 5 {
		t.Fatalf("got %d events, want 5 (text_delta, 2x tool_call, usage, done): %+v", len(events), events)
	}

	td, ok := events[0].(bridle.TextDeltaEvent)
	if !ok || td.Text != "thinking about it" {
		t.Errorf("events[0] = %#v, want TextDeltaEvent{thinking about it}", events[0])
	}

	tc1, ok := events[1].(bridle.ToolCallEvent)
	if !ok || tc1.ID != "call_1" || tc1.Name != "search" {
		t.Errorf("events[1] = %#v, want ToolCallEvent{call_1, search}", events[1])
	}
	tc2, ok := events[2].(bridle.ToolCallEvent)
	if !ok || tc2.ID != "call_2" || tc2.Name != "fetch" {
		t.Errorf("events[2] = %#v, want ToolCallEvent{call_2, fetch} — tool calls must stay in order", events[2])
	}

	usage, ok := events[3].(bridle.UsageEvent)
	if !ok || usage.Usage.Input != 10 || usage.Usage.Output != 5 || usage.Usage.Cached != 2 {
		t.Errorf("events[3] = %#v, want UsageEvent{10,5,2,0}", events[3])
	}

	done, ok := events[4].(bridle.DoneEvent)
	if !ok {
		t.Fatalf("events[4] = %#v, want DoneEvent", events[4])
	}
	if done.StopReason != bridle.StreamStopToolCalls {
		t.Errorf("done.StopReason = %q, want %q (tool_calls present must force this terminal signal)", done.StopReason, bridle.StreamStopToolCalls)
	}
}

// TestStream_DirectAPI_Refusal proves a refusal StopReason maps to
// done{stop_reason:refusal} with no tool calls in play.
func TestStream_DirectAPI_Refusal(t *testing.T) {
	provider := fake.NewProvider(fake.Step{
		StopReason: bridle.StopReasonRefusal,
	})
	h := bridle.NewHarness(provider)

	r := bridle.NewRegistry()
	if err := r.Bind("claude-api", h); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	handle, err := r.Resolve("claude-api/claude-sonnet-5", "")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}

	ch, err := r.Stream(context.Background(), handle, bridle.Request{
		Messages: []bridle.RoleMessage{{Role: bridle.MessageRoleUser, Content: "do something unsafe"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainStream(ch)

	last := events[len(events)-1]
	done, ok := last.(bridle.DoneEvent)
	if !ok {
		t.Fatalf("last event = %#v, want DoneEvent", last)
	}
	if done.StopReason != bridle.StreamStopRefusal {
		t.Errorf("done.StopReason = %q, want %q", done.StopReason, bridle.StreamStopRefusal)
	}
}

// TestStream_ClaudeCode_EventSequence drives the RunTurn (subprocess-
// stream) path via a fake.SubprocessProvider and asserts the same
// normalized sequence shape, proving Stream dispatches claude-code
// through RunTurn (not RunStep) and still produces assembled,
// in-order tool calls.
func TestStream_ClaudeCode_EventSequence(t *testing.T) {
	provider := fake.NewSubprocessProvider(fake.SubprocessStep{
		Text: "checking",
		ToolCalls: []bridle.ToolCallStart{
			{ID: "call_1", Name: "read_file", Args: json.RawMessage(`{"path":"a"}`)},
			{ID: "call_2", Name: "read_file", Args: json.RawMessage(`{"path":"b"}`)},
		},
		ToolResults: []bridle.ToolCallResult{
			{ID: "call_1", Result: json.RawMessage(`"contents-a"`)},
			{ID: "call_2", Result: json.RawMessage(`"contents-b"`)},
		},
	})
	h := bridle.NewHarness(provider)

	r := bridle.NewRegistry()
	if err := r.Bind("claude-code", h); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	handle, err := r.Resolve("claude-code/claude-sonnet-5", "")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}

	ch, err := r.Stream(context.Background(), handle, bridle.Request{
		Messages: []bridle.RoleMessage{{Role: bridle.MessageRoleUser, Content: "read both files"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainStream(ch)

	if len(events) != 5 {
		t.Fatalf("got %d events, want 5 (text_delta, 2x tool_call, usage, done): %+v", len(events), events)
	}
	if td, ok := events[0].(bridle.TextDeltaEvent); !ok || td.Text != "checking" {
		t.Errorf("events[0] = %#v, want TextDeltaEvent{checking}", events[0])
	}
	tc1, ok := events[1].(bridle.ToolCallEvent)
	if !ok || tc1.ID != "call_1" {
		t.Errorf("events[1] = %#v, want ToolCallEvent{call_1, ...}", events[1])
	}
	tc2, ok := events[2].(bridle.ToolCallEvent)
	if !ok || tc2.ID != "call_2" {
		t.Errorf("events[2] = %#v, want ToolCallEvent{call_2, ...} — order preserved", events[2])
	}
	if _, ok := events[3].(bridle.UsageEvent); !ok {
		t.Errorf("events[3] = %#v, want UsageEvent", events[3])
	}
	done, ok := events[4].(bridle.DoneEvent)
	if !ok || done.StopReason != bridle.StreamStopToolCalls {
		t.Errorf("events[4] = %#v, want DoneEvent{tool_calls}", events[4])
	}
}

// TestStream_UnboundLane errors immediately rather than returning a
// channel that just closes empty.
func TestStream_UnboundLane(t *testing.T) {
	r := bridle.NewRegistry()
	handle := bridle.ModelHandle{Lane: "claude-api", Provider: bridle.ProviderClaude, Model: "claude-sonnet-5"}
	_, err := r.Stream(context.Background(), handle, bridle.Request{})
	if err == nil {
		t.Fatal("expected an error streaming against an unbound lane, got nil")
	}
}

// TestStream_RecoversFromProviderPanic proves Stream's spawned goroutine
// has its own recover boundary: a provider panic (here, inside RunStep's
// underlying RunTurn call, on the direct-api dispatch path) must yield an
// ErrorEvent and a closed channel, NOT crash the process. The test
// surviving to completion IS the proof — an unrecovered panic would kill
// the test binary, not just fail an assertion.
func TestStream_RecoversFromProviderPanic(t *testing.T) {
	provider := fake.NewProvider(fake.Step{Panic: true})
	h := bridle.NewHarness(provider)

	r := bridle.NewRegistry()
	if err := r.Bind("claude-api", h); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	handle, err := r.Resolve("claude-api/claude-sonnet-5", "")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}

	ch, err := r.Stream(context.Background(), handle, bridle.Request{
		Messages: []bridle.RoleMessage{{Role: bridle.MessageRoleUser, Content: "trigger the panic"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainStream(ch)

	if len(events) != 1 {
		t.Fatalf("got %d events, want 1 (a single ErrorEvent, no partial tool_call/usage/done): %+v", len(events), events)
	}
	errEv, ok := events[0].(bridle.ErrorEvent)
	if !ok {
		t.Fatalf("events[0] = %#v, want ErrorEvent", events[0])
	}
	if errEv.Err == nil {
		t.Errorf("ErrorEvent.Err is nil, want the recovered panic wrapped as an error")
	}
	// Channel must actually be closed (drainStream only returns once
	// closed, but assert explicitly too — a second receive must report
	// !ok, not block).
	if _, stillOpen := <-ch; stillOpen {
		t.Error("channel still open after the terminal ErrorEvent")
	}
}

// TestClassifyStreamError_MatchesProviderErrorClass is the sync-guard
// finding #7: stream.go's ClassifyStreamError duplicates (but cannot
// call, per import-cycle — see its doc comment) internal/normalize.
// ProviderErrorClass's switch. Nothing else keeps the two in sync, so
// this test iterates every ProviderErrorKind and asserts the two
// mappings agree — the guard against silent drift when T3 adds new
// kinds/classes.
func TestClassifyStreamError_MatchesProviderErrorClass(t *testing.T) {
	kinds := []bridle.ProviderErrorKind{
		bridle.ProviderErrorAuthFailed,
		bridle.ProviderErrorRateLimit,
		bridle.ProviderErrorServerError,
		bridle.ProviderErrorNetworkError,
		bridle.ProviderErrorTimeout,
		bridle.ProviderErrorTLSError,
		bridle.ProviderErrorConfig,
		bridle.ProviderErrorCrash,
		bridle.ProviderErrorSubprocessExit,
	}
	for _, kind := range kinds {
		pe := &bridle.ProviderError{Kind: kind}
		got := bridle.ClassifyStreamError(pe)
		want := normalize.ProviderErrorClass(kind)
		if got != want {
			t.Errorf("ClassifyStreamError(kind=%q) = %q, want %q (normalize.ProviderErrorClass) — the two switches have drifted", kind, got, want)
		}
	}
}
