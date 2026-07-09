package toolrunner

// Local Phase-0 probe (uncommitted): does Ornith drive the bridle
// native-API ToolRunner loop as cleanly as DeepSeek did?

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	openai "github.com/CarriedWorldUniverse/bridle/provider/openai"
)

func TestE2EOrnithToolChain(t *testing.T) {
	base := os.Getenv("BRIDLE_E2E_ORNITH_BASE")
	if base == "" {
		t.Skip("set BRIDLE_E2E_ORNITH_BASE to run the live Ornith chain")
	}

	runner, err := New(Config{WorkDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	prov := openai.NewWithBaseURL("dummy", base)
	h := bridle.NewHarness(prov)

	// Reasoning model: allow generous wall time.
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()

	req := bridle.TurnRequest{
		AspectID: "phase0-ornith-probe",
		Provider: bridle.ProviderOpenAI,
		Model:    "ornith",
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
	var names []string
	for _, tc := range res.ToolCalls {
		names = append(names, tc.Name)
	}
	t.Logf("stop=%s steps=%d tools=%d calls=%v final=%q", res.StopReason, res.StepCount, len(res.ToolCalls), names, res.FinalText)

	if len(res.ToolCalls) < 2 {
		t.Fatalf("expected a multi-step tool chain, got tools=%v", names)
	}
	if !strings.Contains(res.FinalText, "BRIDLE_OK") {
		t.Errorf("final text should report the file content; got %q", res.FinalText)
	}
}
