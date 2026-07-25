package claudesdk_test

import (
	"context"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
	"github.com/CarriedWorldUniverse/bridle/provider/claudesdk"
)

// TestClaudeSDK_RateLimitEvent_SurfacesAsABridleEvent proves the whole
// wire round trip end-to-end via a fake sidecar: a rate_limit_event line
// on the wire must reach the caller as a bridle.RateLimit event with every
// field carried through, and must not affect the turn's own outcome.
func TestClaudeSDK_RateLimitEvent_SurfacesAsABridleEvent(t *testing.T) {
	sidecar := writeFakeSidecar(t, `
echo '{"type":"rate_limit_event","rate_limit_status":"allowed_warning","rate_limit_type":"five_hour","rate_limit_utilization":82,"rate_limit_resets_at_ms":1780000000000,"rate_limit_using_overage":false}'
echo '{"type":"text_delta","text":"still working"}'
echo '{"type":"usage","input_tokens":5,"output_tokens":3}'
echo '{"type":"done","stop_reason":"end_turn","session_id":"sess-1","model":"claude-fake"}'
`)

	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}
	runner := fake.NewToolRunner(nil)

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "claude-fake",
		UserMessage: "hi",
	}, runner, sink)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if result.FinalText != "still working" {
		t.Errorf("a rate_limit_event line disrupted the turn's own text: FinalText = %q", result.FinalText)
	}

	var got *bridle.RateLimit
	for _, ev := range sink.Events {
		if rl, ok := ev.(bridle.RateLimit); ok {
			rl := rl
			got = &rl
		}
	}
	if got == nil {
		t.Fatal("no bridle.RateLimit event reached the sink — the wire event was dropped somewhere in the pipe")
	}
	if got.Status != "allowed_warning" {
		t.Errorf("Status = %q; want allowed_warning", got.Status)
	}
	if got.WindowType != "five_hour" {
		t.Errorf("WindowType = %q; want five_hour", got.WindowType)
	}
	if got.Utilization != 82 {
		t.Errorf("Utilization = %d; want 82", got.Utilization)
	}
	if got.UsingOverage {
		t.Error("UsingOverage = true; the fixture said false")
	}
	if got.ResetsAt.UnixMilli() != 1780000000000 {
		t.Errorf("ResetsAt = %v (unix ms %d); want unix ms 1780000000000", got.ResetsAt, got.ResetsAt.UnixMilli())
	}
	if got.TS.IsZero() {
		t.Error("TS was not stamped")
	}
}

// A missing rate_limit_resets_at_ms (the field omitted entirely, the
// common case for a provider that doesn't know a reset time) must produce
// the zero Time, not a spurious 1970-01-01 from unix-ms(0).
func TestClaudeSDK_RateLimitEvent_MissingResetsAtIsZeroTime(t *testing.T) {
	sidecar := writeFakeSidecar(t, `
echo '{"type":"rate_limit_event","rate_limit_status":"allowed","rate_limit_type":"seven_day","rate_limit_utilization":10}'
echo '{"type":"text_delta","text":"ok"}'
echo '{"type":"usage","input_tokens":1,"output_tokens":1}'
echo '{"type":"done","stop_reason":"end_turn","session_id":"sess-1","model":"claude-fake"}'
`)

	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}
	runner := fake.NewToolRunner(nil)

	if _, err := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "claude-fake", UserMessage: "hi"}, runner, sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	var got *bridle.RateLimit
	for _, ev := range sink.Events {
		if rl, ok := ev.(bridle.RateLimit); ok {
			rl := rl
			got = &rl
		}
	}
	if got == nil {
		t.Fatal("no bridle.RateLimit event reached the sink")
	}
	if !got.ResetsAt.IsZero() {
		t.Errorf("ResetsAt = %v; want the zero Time when the wire omitted it, not epoch", got.ResetsAt)
	}
}

// A turn that never mentions rate limits (the overwhelmingly common case
// today — API-key/Bedrock/Vertex sessions, or a claude.ai session between
// window-boundary readings) must not emit a spurious RateLimit event.
func TestClaudeSDK_NoRateLimitEvent_WhenSidecarNeverSendsOne(t *testing.T) {
	sidecar := writeFakeSidecar(t, `
echo '{"type":"text_delta","text":"ok"}'
echo '{"type":"usage","input_tokens":1,"output_tokens":1}'
echo '{"type":"done","stop_reason":"end_turn","session_id":"sess-1","model":"claude-fake"}'
`)

	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}
	runner := fake.NewToolRunner(nil)

	if _, err := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "claude-fake", UserMessage: "hi"}, runner, sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	for _, ev := range sink.Events {
		if _, ok := ev.(bridle.RateLimit); ok {
			t.Fatal("a bridle.RateLimit event fired when the sidecar never sent rate_limit_event")
		}
	}
}

// The assistant-error "rate_limit" CLASS (a turn actually failing) and
// the rate_limit_event MESSAGE (a usage reading) are unrelated wire
// shapes that happen to share a word. One firing must not produce or
// suppress the other.
func TestClaudeSDK_RateLimitErrorClass_DoesNotProduceARateLimitEvent(t *testing.T) {
	sidecar := writeFakeSidecar(t, `
echo '{"type":"error","class":"rate_limit","message":"rate limited"}'
`)

	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}
	runner := fake.NewToolRunner(nil)

	_, err := h.RunTurn(context.Background(), bridle.TurnRequest{Model: "claude-fake", UserMessage: "hi"}, runner, sink)
	if err == nil {
		t.Fatal("an error-class rate_limit line did not surface as a turn error")
	}
	for _, ev := range sink.Events {
		if _, ok := ev.(bridle.RateLimit); ok {
			t.Fatal("the error-class \"rate_limit\" produced a bridle.RateLimit event — these are unrelated wire shapes")
		}
	}
}
