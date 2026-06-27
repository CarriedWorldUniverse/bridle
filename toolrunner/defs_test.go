package toolrunner

import (
	"encoding/json"
	"testing"
)

func TestDefsCoverAllTools(t *testing.T) {
	want := []string{"bash", "read", "write", "edit", "glob", "grep", "web_fetch", "web_extract"}
	defs := Defs()
	got := map[string]bool{}
	for _, d := range defs {
		got[d.Name] = true
		if len(d.InputSchema) == 0 {
			t.Errorf("%s: empty schema", d.Name)
		}
		var js map[string]any
		if err := json.Unmarshal(d.InputSchema, &js); err != nil {
			t.Errorf("%s: schema not valid JSON: %v", d.Name, err)
		}
	}
	for _, w := range want {
		if !got[w] {
			t.Errorf("missing tool def %q", w)
		}
	}
}
