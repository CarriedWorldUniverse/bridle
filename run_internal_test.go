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
	// merged surface is sorted by Name so the serialized tool block is
	// byte-stable across turns regardless of explicit/MCP listing order.
	if merged[0].Name != "list_files" || merged[1].Name != "send_chat" {
		t.Errorf("merged order = %v; want sorted by name", []string{merged[0].Name, merged[1].Name})
	}
	if merged[0].Description != "list" {
		t.Errorf("mcp tool description lost: %+v", merged[0])
	}
}

func TestMergeToolSurface_SortedRegardlessOfInputOrder(t *testing.T) {
	explicit := []ToolDef{
		{Name: "zeta", InputSchema: json.RawMessage(`{}`)},
		{Name: "alpha", InputSchema: json.RawMessage(`{}`)},
	}
	mcpTools := []mcpclient.ToolDef{
		{Name: "mike", InputSchema: json.RawMessage(`{}`)},
		{Name: "bravo", InputSchema: json.RawMessage(`{}`)},
	}

	merged, err := mergeToolSurface(explicit, mcpTools)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"alpha", "bravo", "mike", "zeta"}
	got := make([]string, len(merged))
	for i, td := range merged {
		got[i] = td.Name
	}
	if len(got) != len(want) {
		t.Fatalf("merged len = %d; want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("merged order = %v; want %v", got, want)
		}
	}

	// Feeding the same tools in reverse input order must yield the same
	// sorted output — the whole point of the sort.
	reversedExplicit := []ToolDef{explicit[1], explicit[0]}
	reversedMCP := []mcpclient.ToolDef{mcpTools[1], mcpTools[0]}
	merged2, err := mergeToolSurface(reversedExplicit, reversedMCP)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i := range want {
		if merged2[i].Name != want[i] {
			t.Errorf("reversed-input merged order = %v; want %v", merged2, want)
		}
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
