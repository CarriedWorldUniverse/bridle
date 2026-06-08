//go:build integration

package claudepty

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// collectSink accumulates ModelChunk text emitted during a turn.
type collectSink struct{ b strings.Builder }

func (s *collectSink) Emit(e bridle.Event) {
	if mc, ok := e.(bridle.ModelChunk); ok {
		s.b.WriteString(mc.Text)
	}
}

// TestProvider_LiveTurn_RealClaude is the e2e for the path plumb now uses:
// bridle's claudepty.Provider → acp-claude-pty → real claude. It runs one
// turn in a fresh spawn dir (relying on the binary's default
// --accept-workspace-trust to get past the folder-trust dialog) and asserts a
// non-empty reply. Gated on a real claude + the binary path:
//
//	ACP_CLAUDE_PTY_BIN=/path/to/acp-claude-pty \
//	  go test -tags=integration ./provider/claudepty/ -run TestProvider_LiveTurn -v
func TestProvider_LiveTurn_RealClaude(t *testing.T) {
	if _, err := exec.LookPath("claude"); err != nil {
		t.Skip("claude not on PATH")
	}
	bin := os.Getenv("ACP_CLAUDE_PTY_BIN")
	if bin == "" {
		t.Skip("set ACP_CLAUDE_PTY_BIN to the built acp-claude-pty binary")
	}

	p := New()
	p.BinaryPath = bin
	t.Cleanup(func() { _ = p.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	sink := &collectSink{}
	res, err := p.RunTurn(ctx, bridle.ProviderRequest{
		AspectID: "plumb-e2e",
		Model:    "claude-opus-4-7",
		Cwd:      t.TempDir(),
		Messages: []bridle.ProviderMessage{
			{Role: "user", Content: "Reply with exactly the single word: PONG"},
		},
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn (bridle -> acp-claude-pty -> claude): %v", err)
	}

	t.Logf("FinalText: %q", res.FinalText)
	if strings.TrimSpace(res.FinalText) == "" {
		t.Fatalf("turn returned empty FinalText (sink: %q)", sink.b.String())
	}
	if !strings.Contains(strings.ToUpper(res.FinalText), "PONG") {
		t.Logf("note: reply did not contain PONG, but a turn completed cleanly: %q", res.FinalText)
	}
}
