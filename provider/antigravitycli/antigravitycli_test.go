package antigravitycli

import (
	"context"
	"encoding/json"
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
	if !caps.SupportsAfterToolCall {
		t.Error("SupportsAfterToolCall should be true")
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
		"--output-format\x00stream-json",
		"--dangerously-skip-permissions",
		"--model\x00gemini-2.0-flash",
		"--add-dir\x00/mywork",
		"--log-file\x00/tmp/antigravity.log",
	} {
		if !strings.Contains(args, want) {
			t.Fatalf("args missing %q in %q", want, args)
		}
	}

	req.Session = bridle.SessionHandle{ID: "session-123"}
	args = strings.Join(p.buildCLIArgs(req), "\x00")
	if !strings.Contains(args, "--conversation\x00session-123") {
		t.Fatalf("resume args missing conversation: %q", args)
	}
}

func TestParseStream_TextOnly(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"init","session_id":"sess-1","model":"model-1"}`,
		`{"type":"message","role":"assistant","content":"hello"}`,
		`{"type":"result","status":"success","stats":{"input_tokens":10,"output_tokens":5}}`,
	}, "\n")

	sink := &fake.SliceEventSink{}
	result, err := parseStream(strings.NewReader(input), sink)
	if err != nil {
		t.Fatalf("parseStream error: %v", err)
	}
	if result.FinalText != "hello" {
		t.Fatalf("FinalText = %q, want hello", result.FinalText)
	}
	if result.StopReason != bridle.StopReasonModelDone {
		t.Fatalf("StopReason = %q, want model_done", result.StopReason)
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 {
		t.Fatalf("Usage = %+v", result.Usage)
	}
	if len(result.SessionDelta) != 2 {
		t.Fatalf("SessionDelta len = %d, want 2", len(result.SessionDelta))
	}
}

func TestParseStream_ToolCall(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"init","session_id":"sess-1","model":"model-1"}`,
		`{"type":"tool_use","tool_name":"get_weather","tool_id":"call-1","parameters":{"location":"Seattle"}}`,
		`{"type":"tool_result","tool_id":"call-1","status":"success","result":{"temp":65}}`,
		`{"type":"result","status":"success","stats":{"input_tokens":15,"output_tokens":20}}`,
	}, "\n")

	sink := &fake.SliceEventSink{}
	result, err := parseStream(strings.NewReader(input), sink)
	if err != nil {
		t.Fatalf("parseStream error: %v", err)
	}
	if result.StepCount != 1 {
		t.Fatalf("StepCount = %d, want 1", result.StepCount)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(result.ToolCalls))
	}
	call := result.ToolCalls[0]
	if call.ID != "call-1" || call.Name != "get_weather" {
		t.Fatalf("ToolCall = %+v", call)
	}
	var args map[string]string
	if err := json.Unmarshal(call.Args, &args); err != nil {
		t.Fatalf("call args unmarshal: %v", err)
	}
	if args["location"] != "Seattle" {
		t.Fatalf("location arg = %q", args["location"])
	}

	var sawStart, sawResult, sawBoundary bool
	for _, ev := range sink.Events {
		switch ev.(type) {
		case bridle.ToolCallStart:
			sawStart = true
		case bridle.ToolCallResult:
			sawResult = true
		case bridle.StepBoundary:
			sawBoundary = true
		}
	}
	if !sawStart || !sawResult || !sawBoundary {
		t.Fatalf("events start=%v result=%v boundary=%v", sawStart, sawResult, sawBoundary)
	}
}

func TestRoundTripLive(t *testing.T) {
	if os.Getenv("BRIDLE_LIVE_ANTIGRAVITY") != "1" {
		t.Skip("set BRIDLE_LIVE_ANTIGRAVITY=1 to run live Antigravity CLI test")
	}
	if _, err := exec.LookPath("antigravity"); err != nil {
		t.Skip("antigravity CLI not on PATH")
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
