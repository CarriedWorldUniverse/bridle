package bridle

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/CarriedWorldUniverse/bridle/internal/mcpclient"
)

// Internal tests for unexported run.go helpers that production depends
// on. Lives in package bridle (not bridle_test) so it can call the
// unexported functions directly.

func TestMergeToolSurface_NoCollision(t *testing.T) {
	explicit := []ToolDef{{Name: "send_chat", Description: "send", InputSchema: json.RawMessage(`{}`)}}
	mcpTools := []mcpclient.ToolDef{{Name: "list_files", Description: "list", InputSchema: json.RawMessage(`{}`)}}

	merged, err := mergeToolSurface(explicit, mcpTools)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(merged) != 2 {
		t.Fatalf("merged len = %d; want 2", len(merged))
	}
	if merged[0].Name != "send_chat" || merged[1].Name != "list_files" {
		t.Errorf("merged order = %v; want explicit-first", []string{merged[0].Name, merged[1].Name})
	}
	if merged[1].Description != "list" {
		t.Errorf("mcp tool description lost: %+v", merged[1])
	}
}

func TestMergeToolSurface_Collision(t *testing.T) {
	explicit := []ToolDef{{Name: "collide", InputSchema: json.RawMessage(`{}`)}}
	mcpTools := []mcpclient.ToolDef{{Name: "collide", InputSchema: json.RawMessage(`{}`)}}

	_, err := mergeToolSurface(explicit, mcpTools)
	if !errors.Is(err, ErrToolNameCollision) {
		t.Errorf("err = %v; want ErrToolNameCollision", err)
	}
}

func TestMergeToolSurface_NilInputs(t *testing.T) {
	merged, err := mergeToolSurface(nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(merged) != 0 {
		t.Errorf("merged len = %d; want 0", len(merged))
	}
}

func TestMergeToolSurface_ExplicitOnly(t *testing.T) {
	explicit := []ToolDef{{Name: "only", InputSchema: json.RawMessage(`{}`)}}
	merged, err := mergeToolSurface(explicit, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(merged) != 1 || merged[0].Name != "only" {
		t.Errorf("merged = %v; want [only]", merged)
	}
}
