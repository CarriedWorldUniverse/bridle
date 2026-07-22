package claudesdk_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
	"github.com/CarriedWorldUniverse/bridle/provider/claudesdk"
)

// TestClaudeSDK_Effort_ReachesInitLine pins that ProviderRequest.Effort
// (the agora reasoning-effort ladder, agora-spec-bridle §3) is present
// verbatim on the "effort" key of the init line bridle sends the
// sidecar's stdin — the wire-level contract wire.go/wire.ts share
// (bridle-agentsdk-spec.md §5). The fake sidecar echoes the raw line it
// read back to a file so the test can decode the ACTUAL bytes bridle
// put on the wire, not a mock of them.
func TestClaudeSDK_Effort_ReachesInitLine(t *testing.T) {
	initFile := filepath.Join(t.TempDir(), "init.jsonl")
	sidecar := writeFakeSidecar(t, `
echo "$init_line" > "`+initFile+`"
echo '{"type":"done","stop_reason":"end_turn"}'
`)

	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "claude-fake",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		Effort:   "xhigh",
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	raw, err := os.ReadFile(initFile)
	if err != nil {
		t.Fatalf("read captured init line: %v", err)
	}
	var init map[string]any
	if err := json.Unmarshal(raw, &init); err != nil {
		t.Fatalf("decode init line %q: %v", raw, err)
	}
	if got := init["effort"]; got != "xhigh" {
		t.Errorf("init.effort = %v, want %q", got, "xhigh")
	}
}

// TestClaudeSDK_Effort_EmptyOmittedFromInitLine pins the provider-
// default posture: an unset Effort must not put an "effort" key on the
// wire at all (wire.go's `json:"effort,omitempty"`), matching every
// other optional field's back-compat contract.
func TestClaudeSDK_Effort_EmptyOmittedFromInitLine(t *testing.T) {
	initFile := filepath.Join(t.TempDir(), "init.jsonl")
	sidecar := writeFakeSidecar(t, `
echo "$init_line" > "`+initFile+`"
echo '{"type":"done","stop_reason":"end_turn"}'
`)

	p := &claudesdk.Provider{SidecarPath: sidecar, Mode: claudesdk.ModeFunnel}
	sink := &fake.SliceEventSink{}

	_, err := p.RunTurn(context.Background(), bridle.ProviderRequest{
		Model:    "claude-fake",
		Messages: []bridle.ProviderMessage{{Role: "user", Content: "hi"}},
		// Effort deliberately left empty.
	}, sink)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	raw, err := os.ReadFile(initFile)
	if err != nil {
		t.Fatalf("read captured init line: %v", err)
	}
	var init map[string]any
	if err := json.Unmarshal(raw, &init); err != nil {
		t.Fatalf("decode init line %q: %v", raw, err)
	}
	if _, present := init["effort"]; present {
		t.Errorf("init.effort should be absent when Effort is empty; got %v", init["effort"])
	}
}
