package toolrunner

// Phase-0 probe 2 (uncommitted): loop-closing. Ornith gets a buggy
// function + failing test and must fix until green — the exact
// weakness the admin-htmx eval flagged (claims done without verifying).

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	openai "github.com/CarriedWorldUniverse/bridle/provider/openai"
)

const buggySrc = `package clamp

// Clamp returns v limited to the inclusive range [lo, hi].
func Clamp(v, lo, hi int) int {
	if v < lo {
		return hi // BUG: should return lo
	}
	if v > hi {
		return lo // BUG: should return hi
	}
	return v
}
`

const clampTest = `package clamp

import "testing"

func TestClamp(t *testing.T) {
	cases := []struct{ v, lo, hi, want int }{
		{5, 0, 10, 5},
		{-3, 0, 10, 0},
		{42, 0, 10, 10},
	}
	for _, c := range cases {
		if got := Clamp(c.v, c.lo, c.hi); got != c.want {
			t.Errorf("Clamp(%d,%d,%d)=%d want %d", c.v, c.lo, c.hi, got, c.want)
		}
	}
}
`

const goMod = "module clampprobe\n\ngo 1.24\n"

func TestE2EOrnithFixUntilGreen(t *testing.T) {
	base := os.Getenv("BRIDLE_E2E_ORNITH_BASE")
	if base == "" {
		t.Skip("set BRIDLE_E2E_ORNITH_BASE to run the live Ornith fix chain")
	}

	dir := t.TempDir()
	for name, content := range map[string]string{
		"clamp.go": buggySrc, "clamp_test.go": clampTest, "go.mod": goMod,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	runner, err := New(Config{WorkDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	prov := openai.NewWithBaseURL("dummy", base)
	h := bridle.NewHarness(prov)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	req := bridle.TurnRequest{
		AspectID: "phase0-ornith-fix",
		Provider: bridle.ProviderOpenAI,
		Model:    "ornith",
		MaxSteps: 14,
		Tools:    Defs(),
		UserMessage: "This directory contains a Go package with a failing test. " +
			"Run `go test ./...` to see the failure, find and fix the bug in the source " +
			"(do NOT change the test), and re-run the tests. Only report done when " +
			"`go test ./...` actually passes; include the final test output in your reply.",
	}

	res, err := h.RunTurn(ctx, req, runner, &sink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	var names []string
	for _, tc := range res.ToolCalls {
		names = append(names, tc.Name)
	}
	t.Logf("stop=%s steps=%d tools=%d calls=%v", res.StopReason, res.StepCount, len(res.ToolCalls), names)
	t.Logf("final=%q", res.FinalText)

	// Independent ground truth: did it actually leave the tests green?
	fixed, err := os.ReadFile(filepath.Join(dir, "clamp.go"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(fixed), "return hi // BUG") || strings.Contains(string(fixed), "return lo // BUG") {
		t.Errorf("bug markers still present — source not actually fixed:\n%s", fixed)
	}
	if !strings.Contains(strings.ToLower(res.FinalText), "pass") && !strings.Contains(res.FinalText, "ok") {
		t.Errorf("final text does not report passing tests: %q", res.FinalText)
	}
}
