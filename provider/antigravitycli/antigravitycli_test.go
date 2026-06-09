package antigravitycli

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

func TestCapabilities(t *testing.T) {
	p := New()
	if p.Name() != bridle.ProviderAntigravityCLI {
		t.Fatalf("Name = %q, want %q", p.Name(), bridle.ProviderAntigravityCLI)
	}
	caps := p.Capabilities()
	if caps.Category != bridle.CategorySubprocessStream {
		t.Errorf("Category = %q, want subprocess-stream", caps.Category)
	}
	if caps.SupportsCustomTools {
		t.Error("SupportsCustomTools should be false")
	}
	if caps.SupportsBeforeToolCall {
		t.Error("SupportsBeforeToolCall should be false")
	}
	// agy has no structured-output mode, so no per-tool-call events are emitted.
	if caps.SupportsAfterToolCall {
		t.Error("SupportsAfterToolCall should be false (agy streams no tool events)")
	}
	if caps.SupportsMCP {
		t.Error("SupportsMCP should be false")
	}
}

func TestBuildCLIArgs_NewAndResume(t *testing.T) {
	p := New()
	p.ExtraArgs = []string{"--log-file", "/tmp/antigravity.log"}

	req := bridle.ProviderRequest{
		Model: "gemini-2.0-flash",
		Cwd:   "/mywork",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "hello"},
		},
	}

	args := strings.Join(p.buildCLIArgs(req), "\x00")
	for _, want := range []string{
		"-p\x00hello",
		"--dangerously-skip-permissions",
		"--model\x00gemini-2.0-flash",
		"--add-dir\x00/mywork",
		"--log-file\x00/tmp/antigravity.log",
	} {
		if !strings.Contains(args, want) {
			t.Fatalf("args missing %q in %q", want, args)
		}
	}
	// agy rejects these — they must NOT be passed (the bug the local test caught).
	for _, forbidden := range []string{"--output-format", "--allowed-tools"} {
		if strings.Contains(args, forbidden) {
			t.Fatalf("args must not contain %q (agy has no such flag): %q", forbidden, args)
		}
	}

	req.Session = bridle.SessionHandle{ID: "session-123"}
	args = strings.Join(p.buildCLIArgs(req), "\x00")
	if !strings.Contains(args, "--conversation\x00session-123") {
		t.Fatalf("resume args missing conversation: %q", args)
	}
}

func TestCapturePlainText(t *testing.T) {
	const out = "pipeline check ok\n"

	sink := &fake.SliceEventSink{}
	result, err := capturePlainText(strings.NewReader(out), sink)
	if err != nil {
		t.Fatalf("capturePlainText error: %v", err)
	}
	if result.FinalText != "pipeline check ok" {
		t.Fatalf("FinalText = %q, want %q", result.FinalText, "pipeline check ok")
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Fatalf("StopReason = %q, want model_done", result.StopReason)
	}
	if len(result.SessionDelta) != 1 {
		t.Fatalf("SessionDelta len = %d, want 1", len(result.SessionDelta))
	}
	if len(sink.Events) == 0 {
		t.Fatal("expected at least one emitted event (assistant text)")
	}
}

func TestCapturePlainText_Empty(t *testing.T) {
	sink := &fake.SliceEventSink{}
	result, err := capturePlainText(strings.NewReader("   \n"), sink)
	if err != nil {
		t.Fatalf("capturePlainText error: %v", err)
	}
	if result.FinalText != "" {
		t.Fatalf("FinalText = %q, want empty", result.FinalText)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Fatalf("StopReason = %q, want model_done", result.StopReason)
	}
}

func TestCapturePlainText_StripsLeadingStaleConversationWarning(t *testing.T) {
	const out = "Warning: conversation \"stale-session\" not found.\nactual reply\n"

	sink := &fake.SliceEventSink{}
	result, err := capturePlainText(strings.NewReader(out), sink)
	if err != nil {
		t.Fatalf("capturePlainText error: %v", err)
	}
	if result.FinalText != "actual reply" {
		t.Fatalf("FinalText = %q, want %q", result.FinalText, "actual reply")
	}
	if strings.Contains(result.FinalText, "Warning: conversation") {
		t.Fatalf("FinalText leaked stale conversation warning: %q", result.FinalText)
	}
	if len(result.SessionDelta) != 0 {
		t.Fatalf("SessionDelta len = %d, want 0 for stale conversation", len(result.SessionDelta))
	}
	if len(sink.Events) == 0 {
		t.Fatal("expected clean assistant text event")
	}
}

func TestCapturePlainText_PureStaleConversationWarningYieldsEmpty(t *testing.T) {
	const out = "Warning: conversation \"stale-session\" not found."

	sink := &fake.SliceEventSink{}
	result, err := capturePlainText(strings.NewReader(out), sink)
	if err != nil {
		t.Fatalf("capturePlainText error: %v", err)
	}
	if result.FinalText != "" {
		t.Fatalf("FinalText = %q, want empty", result.FinalText)
	}
	if len(result.SessionDelta) != 0 {
		t.Fatalf("SessionDelta len = %d, want 0", len(result.SessionDelta))
	}
	if len(sink.Events) != 0 {
		t.Fatalf("events len = %d, want 0", len(sink.Events))
	}
}

func TestCapturePlainText_NormalOutputUntouched(t *testing.T) {
	const out = "normal output\n"

	sink := &fake.SliceEventSink{}
	result, err := capturePlainText(strings.NewReader(out), sink)
	if err != nil {
		t.Fatalf("capturePlainText error: %v", err)
	}
	if result.FinalText != "normal output" {
		t.Fatalf("FinalText = %q, want %q", result.FinalText, "normal output")
	}
	if len(result.SessionDelta) != 1 {
		t.Fatalf("SessionDelta len = %d, want 1", len(result.SessionDelta))
	}
}

func TestRoundTripLive(t *testing.T) {
	if os.Getenv("BRIDLE_LIVE_ANTIGRAVITY") != "1" {
		t.Skip("set BRIDLE_LIVE_ANTIGRAVITY=1 to run live Antigravity CLI test")
	}
	if _, err := exec.LookPath("agy"); err != nil {
		t.Skip("agy CLI not on PATH")
	}

	p := New()
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "gemini-2.0-flash",
		UserMessage: "Reply with exactly the word: PONG",
		MaxSteps:    1,
	}, fake.NewToolRunner(nil), sink)
	if err != nil {
		t.Fatalf("RunTurn error: %v", err)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Fatalf("StopReason = %q, want model_done", result.StopReason)
	}
	if !strings.Contains(result.FinalText, "PONG") {
		t.Fatalf("FinalText = %q, want PONG", result.FinalText)
	}
}
