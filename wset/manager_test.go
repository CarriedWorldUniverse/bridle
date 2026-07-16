package wset

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// driver simulates the harness loop: appends assistant-call/result pairs,
// runs afterToolCall on each result and beforeModelCall per step.
type driver struct {
	t    *testing.T
	m    *Manager
	msgs []bridle.ProviderMessage
	step int
	n    int
}

func newDriver(t *testing.T, cfg ManagerConfig) *driver {
	t.Helper()
	m := &Manager{cfg: cfg, keys: map[string]*keyState{}}
	return &driver{t: t, m: m}
}

// call appends one tool exchange and runs afterToolCall; returns the result
// content AFTER hook mutation.
func (d *driver) call(tool string, args map[string]string, result string) string {
	d.t.Helper()
	d.n++
	d.step++
	id := fmt.Sprintf("c%d", d.n)
	ab, _ := json.Marshal(args)
	d.msgs = append(d.msgs, bridle.ProviderMessage{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{ID: id, Name: tool, Args: ab}}})
	rb, _ := json.Marshal(result)
	out, _, err := d.m.afterToolCall(context.Background(), bridle.AfterToolCallCtx{
		Call:   bridle.ToolCall{ID: id, Name: tool, Args: ab},
		Result: bridle.ToolCallResult{Result: rb},
		Step:   d.step,
	})
	if err != nil {
		d.t.Fatalf("afterToolCall: %v", err)
	}
	var final string
	json.Unmarshal(out.Result.Result, &final)
	d.msgs = append(d.msgs, bridle.ProviderMessage{Role: "tool_result", ToolCallID: id, ToolName: tool, Content: final})
	return final
}

// assemble runs beforeModelCall over the accumulated history.
func (d *driver) assemble() {
	d.t.Helper()
	d.step++
	_, _, err := d.m.beforeModelCall(context.Background(), bridle.BeforeModelCallCtx{
		Step:    d.step,
		Request: &bridle.ProviderRequest{Messages: d.msgs},
	})
	if err != nil {
		d.t.Fatalf("beforeModelCall: %v", err)
	}
}

func (d *driver) result(n int) string { // content of the n-th tool result (1-based call order)
	d.t.Helper()
	id := fmt.Sprintf("c%d", n)
	for _, m := range d.msgs {
		if m.Role == "tool_result" && m.ToolCallID == id {
			return m.Content
		}
	}
	d.t.Fatalf("no result for call %d", n)
	return ""
}

func testCfg(root string) ManagerConfig {
	cfg := DefaultManagerConfig(root)
	cfg.HotSteps = 1
	return cfg
}

func writeFile(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, rel)
	os.MkdirAll(filepath.Dir(p), 0o755)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestSupersessionReadsAndWriteArgs(t *testing.T) {
	root := t.TempDir()
	d := newDriver(t, testCfg(root))
	writeFile(t, root, "a.py", "v1")
	d.call("read_file", map[string]string{"path": "a.py"}, "v1")
	d.call("write_file", map[string]string{"path": "a.py", "content": "v2"}, "ok")
	writeFile(t, root, "a.py", "v2")
	d.assemble()

	if got := d.result(1); !strings.Contains(got, "superseded") {
		t.Fatalf("old read must be superseded: %q", got)
	}
	// live copy is now the write args — they must be intact
	var args map[string]string
	json.Unmarshal(d.msgs[2].ToolCalls[0].Args, &args)
	if args["content"] != "v2" {
		t.Fatalf("live write args must be intact: %v", args)
	}
	// a later read supersedes the write args
	d.call("read_file", map[string]string{"path": "a.py"}, "v2")
	d.assemble()
	json.Unmarshal(d.msgs[2].ToolCalls[0].Args, &args)
	if !strings.Contains(args["content"], "superseded by a later") {
		t.Fatalf("old write args must be rewritten: %v", args)
	}
	if got := d.result(3); got != "v2" {
		t.Fatalf("newest read is the live copy: %q", got)
	}
}

func TestBudgetDemotionAndHotImmunity(t *testing.T) {
	root := t.TempDir()
	cfg := testCfg(root)
	cfg.ContextWindowTokens = 400 // budget = 100 tokens = ~400 chars
	d := newDriver(t, cfg)

	cold := strings.Repeat("c", 300)
	hot := strings.Repeat("h", 300)
	writeFile(t, root, "cold.py", cold)
	writeFile(t, root, "hot.py", hot)
	d.call("read_file", map[string]string{"path": "cold.py"}, cold)
	d.step += 5 // cold.py goes untouched for 5 steps
	d.call("read_file", map[string]string{"path": "hot.py"}, hot)
	d.assemble()

	if got := d.result(1); !strings.Contains(got, "demoted") || !strings.Contains(got, "tracked") {
		t.Fatalf("cold key must be demoted with tracked stub: %q", got)
	}
	if got := d.result(2); got != hot {
		t.Fatalf("hot key must survive demotion: %q", got[:20])
	}
}

func TestReadmitSource1UnstubsOriginalBytes(t *testing.T) {
	root := t.TempDir()
	cfg := testCfg(root)
	cfg.ContextWindowTokens = 400
	d := newDriver(t, cfg)

	content := strings.Repeat("x", 300) + " cold.py marker"
	writeFile(t, root, "cold.py", content)
	d.call("read_file", map[string]string{"path": "cold.py"}, content)
	d.step += 5
	filler := strings.Repeat("f", 300)
	writeFile(t, root, "other.py", filler)
	d.call("read_file", map[string]string{"path": "other.py"}, filler)
	d.assemble()
	if !strings.Contains(d.result(1), "demoted") {
		t.Fatalf("setup: cold.py should be demoted")
	}
	// a command output mentions cold.py -> re-admission source 1 (disk unchanged)
	d.call("run_command", map[string]string{"command": "grep x"}, "match in cold.py line 3")
	d.assemble()
	if got := d.result(1); got != content {
		t.Fatalf("re-admission must restore ORIGINAL bytes, got %q", got[:40])
	}
}

func TestReadmitSource2HarnessReadOnStale(t *testing.T) {
	root := t.TempDir()
	cfg := testCfg(root)
	cfg.ContextWindowTokens = 400
	d := newDriver(t, cfg)

	old := strings.Repeat("o", 300) + " stale.py"
	writeFile(t, root, "stale.py", old)
	d.call("read_file", map[string]string{"path": "stale.py"}, old)
	d.step += 5
	filler := strings.Repeat("f", 300)
	writeFile(t, root, "other.py", filler)
	d.call("read_file", map[string]string{"path": "other.py"}, filler)
	d.assemble() // stale.py demoted

	writeFile(t, root, "stale.py", "FRESH CONTENT") // external modification
	got := d.call("run_command", map[string]string{"command": "make"}, "rebuilt stale.py")
	if !strings.Contains(got, "working set refresh") || !strings.Contains(got, "FRESH CONTENT") {
		t.Fatalf("stale mention must get a harness-read refresh in the triggering result: %q", got)
	}
}

func TestEditRefreshDeliversSpan(t *testing.T) {
	root := t.TempDir()
	cfg := testCfg(root)
	cfg.PartialThreshold = 50 // force partial on the big file
	d := newDriver(t, cfg)

	lines := make([]string, 40)
	for i := range lines {
		lines[i] = fmt.Sprintf("line%02d", i)
	}
	orig := strings.Join(lines, "\n")
	writeFile(t, root, "big.py", orig)
	d.call("read_file", map[string]string{"path": "big.py"}, orig)

	lines[20] = "EDITED"
	writeFile(t, root, "big.py", strings.Join(lines, "\n"))
	got := d.call("edit_file", map[string]string{"path": "big.py"}, "ok")
	if !strings.Contains(got, "EDITED") || !strings.Contains(got, "rest tracked") {
		t.Fatalf("edit must deliver the changed span: %q", got)
	}
	if strings.Contains(got, "line01") {
		t.Fatalf("span must be partial, not the whole file: %q", got)
	}
	d.assemble()
	if r := d.result(1); !strings.Contains(r, "superseded") {
		t.Fatalf("pre-edit read must be superseded after refresh: %q", r)
	}
}

func TestExternalChangeStubsStaleLiveCopy(t *testing.T) {
	root := t.TempDir()
	d := newDriver(t, testCfg(root))
	writeFile(t, root, "a.py", "v1")
	d.call("read_file", map[string]string{"path": "a.py"}, "v1")
	writeFile(t, root, "a.py", "changed-behind-our-back")
	d.call("run_command", map[string]string{"command": "sed -i ..."}, "done")
	d.assemble()
	if got := d.result(1); !strings.Contains(got, "modified since this read") {
		t.Fatalf("stale live copy must be invalidated, got %q", got)
	}
}

func TestEditToolLiveCarrierNotAgedOut(t *testing.T) {
	root := t.TempDir()
	d := newDriver(t, testCfg(root))
	writeFile(t, root, "a.py", "small")
	d.call("edit_file", map[string]string{"path": "a.py"}, "ok") // carries refresh
	d.call("run_command", map[string]string{"command": "1"}, "o1")
	d.call("run_command", map[string]string{"command": "2"}, "o2")
	d.call("run_command", map[string]string{"command": "3"}, "o3")
	d.assemble()
	if got := d.result(1); !strings.Contains(got, "small") {
		t.Fatalf("edit result carrying the live copy must not age out: %q", got)
	}
	if got := d.result(2); !strings.HasPrefix(got, "[tool result evicted") {
		t.Fatalf("oldest plain command must age out: %q", got)
	}
}

func TestAdoptsPreAttachHistory(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a.py", "adopted")
	d := newDriver(t, testCfg(root))
	// history built BEFORE the manager saw any afterToolCall
	ab, _ := json.Marshal(map[string]string{"path": "a.py"})
	d.msgs = append(d.msgs,
		bridle.ProviderMessage{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{ID: "old1", Name: "read_file", Args: ab}}},
		bridle.ProviderMessage{Role: "tool_result", ToolCallID: "old1", Content: "adopted"},
	)
	d.assemble()
	for _, m := range d.msgs {
		if m.Role == "tool_result" && m.Content != "adopted" {
			t.Fatalf("unseen history must be adopted as live, not stubbed: %q", m.Content)
		}
	}
}

func TestManagerAttachDetach(t *testing.T) {
	h := bridle.NewHarness(nil)
	_, detach := AttachManager(h, DefaultManagerConfig(t.TempDir()))
	detach()
}
