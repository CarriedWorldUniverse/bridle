package codexcli

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
	if p.Name() != bridle.ProviderCodexCLI {
		t.Fatalf("Name = %q, want %q", p.Name(), bridle.ProviderCodexCLI)
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
	if !caps.SupportsMCP {
		t.Error("SupportsMCP should be true (MCP wired via --config mcp_servers.*)")
	}
}

func TestNewDefaults(t *testing.T) {
	p := New()
	// Codex aspects run inside the funnel's trust boundary, so codex's
	// internal sandbox is disabled by default (its workspace-write default
	// blocks network / git push).
	if p.Sandbox != "danger-full-access" {
		t.Errorf("Sandbox = %q, want danger-full-access", p.Sandbox)
	}
	if p.ApprovalPolicy != "never" {
		t.Errorf("ApprovalPolicy = %q, want never", p.ApprovalPolicy)
	}
	// And it surfaces in the built args.
	args := strings.Join(p.buildCLIArgs(bridle.ProviderRequest{}), "\x00")
	if !strings.Contains(args, "--sandbox\x00danger-full-access") {
		t.Errorf("args missing --sandbox danger-full-access: %q", args)
	}
}

func TestBuildCLIArgs_NewAndResume(t *testing.T) {
	p := New()
	p.Sandbox = "read-only"
	p.Profile = "worker"
	p.ExtraConfig = []string{`model_reasoning_effort="low"`}

	req := bridle.ProviderRequest{
		Model: "gpt-5-codex",
		Cwd:   "/work",
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "older"},
			{Role: "assistant", Content: "context"},
			{Role: "user", Content: "current"},
		},
	}

	args := strings.Join(p.buildCLIArgs(req), "\x00")
	for _, want := range []string{
		"--profile\x00worker",
		"--sandbox\x00read-only",
		"--ask-for-approval\x00never",
		"--config\x00model_reasoning_effort=\"low\"",
		"--cd\x00/work",
		"exec\x00--json",
		"--skip-git-repo-check",
		"--model\x00gpt-5-codex",
		"current",
	} {
		if !strings.Contains(args, want) {
			t.Fatalf("args missing %q in %q", want, args)
		}
	}

	req.Session = bridle.SessionHandle{ID: "019e8c6d-4415-70b0-942f-de66b23c4f10"}
	args = strings.Join(p.buildCLIArgs(req), "\x00")
	if !strings.Contains(args, "exec\x00resume\x00019e8c6d-4415-70b0-942f-de66b23c4f10\x00--json") {
		t.Fatalf("resume args not shaped correctly: %q", args)
	}

	req.Model = "default"
	args = strings.Join(p.buildCLIArgs(req), "\x00")
	if strings.Contains(args, "--model") {
		t.Fatalf("default model should omit --model: %q", args)
	}
}

func TestParseStream_TextOnly(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"thread.started","thread_id":"019e8c6d-4415-70b0-942f-de66b23c4f10"}`,
		`{"type":"turn.started"}`,
		`{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"hello"}}`,
		`{"type":"turn.completed","usage":{"input_tokens":22238,"cached_input_tokens":20352,"output_tokens":5,"reasoning_output_tokens":0}}`,
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
	if result.Usage.InputTokens != 22238 || result.Usage.CacheReadInputTokens != 20352 || result.Usage.OutputTokens != 5 {
		t.Fatalf("Usage = %+v", result.Usage)
	}
	if len(result.SessionDelta) != 2 {
		t.Fatalf("SessionDelta len = %d, want 2", len(result.SessionDelta))
	}
	norm, err := bridle.ParseSessionEvent(result.SessionDelta[0])
	if err != nil {
		t.Fatalf("ParseSessionEvent error: %v", err)
	}
	if !strings.Contains(norm.Content, "019e8c6d") {
		t.Fatalf("normalized init = %q", norm.Content)
	}
}

func TestParseStream_CommandExecution(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"thread.started","thread_id":"thread-1"}`,
		`{"type":"turn.started"}`,
		`{"type":"item.started","item":{"id":"item_0","type":"command_execution","command":"/bin/zsh -lc pwd","aggregated_output":"","exit_code":null,"status":"in_progress"}}`,
		`{"type":"item.completed","item":{"id":"item_0","type":"command_execution","command":"/bin/zsh -lc pwd","aggregated_output":"/work\n","exit_code":0,"status":"completed"}}`,
		`{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"done"}}`,
		`{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":4}}`,
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
	if call.ID != "item_0" || call.Name != "command_execution" {
		t.Fatalf("ToolCall = %+v", call)
	}
	var args map[string]string
	if err := json.Unmarshal(call.Args, &args); err != nil {
		t.Fatalf("call args unmarshal: %v", err)
	}
	if args["command"] != "/bin/zsh -lc pwd" {
		t.Fatalf("command arg = %q", args["command"])
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
	if os.Getenv("BRIDLE_LIVE_CODEX") != "1" {
		t.Skip("set BRIDLE_LIVE_CODEX=1 to run live Codex CLI test")
	}
	if _, err := exec.LookPath("codex"); err != nil {
		t.Skip("codex CLI not on PATH")
	}

	p := New()
	p.Ephemeral = true
	p.Sandbox = "read-only"
	h := bridle.NewHarness(p)
	sink := &fake.SliceEventSink{}

	result, err := h.RunTurn(context.Background(), bridle.TurnRequest{
		Model:       "default",
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
