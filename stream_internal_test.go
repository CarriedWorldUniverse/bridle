package bridle

import (
	"testing"
)

// Internal tests for unexported stream.go helpers that production
// depends on. Lives in package bridle (not bridle_test) so it can call
// the unexported functions directly.

// TestLowerStreamToProviderRequest_SortsToolsByName proves the
// direct-api Stream path (lowerStreamToProviderRequest) sorts req.Tools
// by name, matching mergeToolSurface's ordering exactly (run.go) — the
// byte-stable-prompt-prefix invariant vLLM prefix caching depends on.
// Stream previously passed req.Tools through verbatim, unsorted.
func TestLowerStreamToProviderRequest_SortsToolsByName(t *testing.T) {
	req := Request{
		Tools: []ToolDef{
			{Name: "zeta"},
			{Name: "alpha"},
			{Name: "mike"},
		},
	}
	preq := lowerStreamToProviderRequest(ModelHandle{Model: "fake-model"}, req)

	want := []string{"alpha", "mike", "zeta"}
	if len(preq.Tools) != len(want) {
		t.Fatalf("Tools len = %d, want %d", len(preq.Tools), len(want))
	}
	for i, name := range want {
		if preq.Tools[i].Name != name {
			t.Errorf("Tools[%d] = %q, want %q (not sorted by name)", i, preq.Tools[i].Name, name)
		}
	}

	// The caller's slice must not be mutated in place.
	if req.Tools[0].Name != "zeta" {
		t.Errorf("caller's req.Tools was mutated: %v", req.Tools)
	}
}
