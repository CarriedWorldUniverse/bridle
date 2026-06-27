package toolrunner

import (
	"encoding/json"
	"testing"
)

func TestGlobAndGrep(t *testing.T) {
	r := newTestRunner(t)
	_, _ = r.runWrite(json.RawMessage(`{"path":"x/one.go","content":"package x\n// TODO: hi\n"}`))
	_, _ = r.runWrite(json.RawMessage(`{"path":"x/two.txt","content":"nothing here"}`))

	g, err := r.runGlob(json.RawMessage(`{"pattern":"**/*.go"}`))
	if err != nil {
		t.Fatal(err)
	}
	var gr globResult
	_ = json.Unmarshal(g, &gr)
	if len(gr.Matches) != 1 || gr.Matches[0] != "x/one.go" {
		t.Fatalf("glob got %+v", gr.Matches)
	}

	gp, err := r.runGrep(json.RawMessage(`{"pattern":"TODO","glob":"**/*.go"}`))
	if err != nil {
		t.Fatal(err)
	}
	var gpr grepResult
	_ = json.Unmarshal(gp, &gpr)
	if len(gpr.Matches) != 1 || gpr.Matches[0].Line != 2 {
		t.Fatalf("grep got %+v", gpr.Matches)
	}
}
