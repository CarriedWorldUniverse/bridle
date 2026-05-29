package toolrunner

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	openai "github.com/CarriedWorldUniverse/bridle/provider/openai"
)

func TestE2EDeepSeekToolChain(t *testing.T) {
	key := os.Getenv("BRIDLE_E2E_OPENAI_KEY")
	if key == "" {
		t.Skip("set BRIDLE_E2E_OPENAI_KEY to run the live DeepSeek chain")
	}
	base := os.Getenv("BRIDLE_E2E_OPENAI_BASE")
	if base == "" {
		base = "https://api.deepseek.com/v1"
	}

	runner, err := New(Config{WorkDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	prov := openai.NewWithBaseURL(key, base)
	h := bridle.NewHarness(prov)

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	req := bridle.TurnRequest{
		AspectID: "e2e-test",
		Provider: bridle.ProviderOpenAI,
		Model:    "deepseek-chat",
		MaxSteps: 8,
		Tools:    Defs(),
		UserMessage: "Using the tools: 1) write a file notes.txt containing exactly the text BRIDLE_OK. " +
			"2) read it back. 3) run the bash command `wc -c < notes.txt`. " +
			"Then reply with the file's content and its byte count.",
	}

	res, err := h.RunTurn(ctx, req, runner, &sink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	t.Logf("stop=%s steps=%d tools=%d final=%q", res.StopReason, res.StepCount, len(res.ToolCalls), res.FinalText)

	// Proof the chain actually executed tools, not just hallucinated.
	var names []string
	for _, tc := range res.ToolCalls {
		names = append(names, tc.Name)
	}
	if len(res.ToolCalls) < 2 {
		t.Fatalf("expected a multi-step tool chain, got tools=%v", names)
	}
	if !strings.Contains(res.FinalText, "BRIDLE_OK") {
		t.Errorf("final text should report the file content; got %q", res.FinalText)
	}
}

type sink struct{}

func (sink) Emit(bridle.Event) {}
