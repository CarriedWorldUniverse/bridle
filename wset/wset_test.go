package wset

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// history builds an alternating assistant-toolcall / tool_result message list
// from (tool, key, content) triples — the shape RunTurn accumulates.
func history(triples ...[3]string) []bridle.ProviderMessage {
	var msgs []bridle.ProviderMessage
	for n, t := range triples {
		id := fmt.Sprintf("call_%d", n)
		var args json.RawMessage
		if t[1] != "" {
			b, _ := json.Marshal(map[string]string{"path": t[1]})
			args = b
		} else {
			args = json.RawMessage(`{}`)
		}
		msgs = append(msgs,
			bridle.ProviderMessage{Role: "assistant", ToolCalls: []bridle.ToolInvocation{{ID: id, Name: t[0], Args: args}}},
			bridle.ProviderMessage{Role: "tool_result", ToolCallID: id, ToolName: t[0], Content: t[2]},
		)
	}
	return msgs
}

func apply(t *testing.T, cfg Config, msgs []bridle.ProviderMessage) []bridle.ProviderMessage {
	t.Helper()
	a := &attachment{cfg: cfg}
	in := bridle.BeforeModelCallCtx{Request: &bridle.ProviderRequest{Messages: msgs}}
	out, action, err := a.beforeModelCall(context.Background(), in)
	if err != nil || action != bridle.HookContinue {
		t.Fatalf("hook: action=%v err=%v", action, err)
	}
	return out.Request.Messages
}

// results extracts tool_result contents in order.
func results(msgs []bridle.ProviderMessage) []string {
	var out []string
	for _, m := range msgs {
		if m.Role == "tool_result" {
			out = append(out, m.Content)
		}
	}
	return out
}

func TestLatestReadPerFileRetained(t *testing.T) {
	msgs := history(
		[3]string{"read_file", "a.py", "A-v1"},
		[3]string{"read_file", "b.py", "B-v1"},
		[3]string{"read_file", "a.py", "A-v2"},
	)
	got := results(apply(t, DefaultConfig(), msgs))
	if !strings.Contains(got[0], `superseded: a newer read_file of "a.py"`) {
		t.Fatalf("old read of a.py must be superseded, got %q", got[0])
	}
	if got[1] != "B-v1" || got[2] != "A-v2" {
		t.Fatalf("latest reads must be retained verbatim, got %q %q", got[1], got[2])
	}
}

func TestOthersAgeOut(t *testing.T) {
	msgs := history(
		[3]string{"run_command", "", "out-1"},
		[3]string{"run_command", "", "out-2"},
		[3]string{"run_command", "", "out-3"},
		[3]string{"read_file", "a.py", "A-v1"},
	)
	got := results(apply(t, DefaultConfig(), msgs)) // KeepOthers=2
	if !strings.HasPrefix(got[0], "[tool result evicted:") {
		t.Fatalf("oldest command output must age out, got %q", got[0])
	}
	if got[1] != "out-2" || got[2] != "out-3" || got[3] != "A-v1" {
		t.Fatalf("window + read must survive, got %v", got[1:])
	}
}

func TestIdempotentAndPrefixStable(t *testing.T) {
	msgs := history(
		[3]string{"read_file", "a.py", "A-v1"},
		[3]string{"run_command", "", "out-1"},
		[3]string{"run_command", "", "out-2"},
		[3]string{"run_command", "", "out-3"},
		[3]string{"read_file", "a.py", "A-v2"},
	)
	once := results(apply(t, DefaultConfig(), msgs))
	twice := results(apply(t, DefaultConfig(), msgs))
	for i := range once {
		if once[i] != twice[i] {
			t.Fatalf("second application must be byte-identical at %d:\n%q\n%q", i, once[i], twice[i])
		}
	}
}

func TestRetainedResultCapped(t *testing.T) {
	big := strings.Repeat("x", 200)
	cfg := DefaultConfig()
	cfg.MaxRetainBytes = 100
	msgs := history(
		[3]string{"read_file", "a.py", big},
		[3]string{"run_command", "", big},
	)
	got := results(apply(t, cfg, msgs))
	for i, g := range got {
		if len(g) >= 200 || !strings.Contains(g, "…result truncated") {
			t.Fatalf("retained result %d must be capped with marker, got %d bytes", i, len(g))
		}
	}
	// and capping is idempotent (byte-identical on re-application)
	again := results(apply(t, cfg, msgs))
	for i := range got {
		if got[i] != again[i] {
			t.Fatalf("truncation must be idempotent at %d", i)
		}
	}
}

func TestUnkeyedReadFallsBackToOthers(t *testing.T) {
	// read_file with no path arg cannot join the working set — it must age
	// out with the others rather than be retained forever.
	msgs := history(
		[3]string{"read_file", "", "mystery-1"},
		[3]string{"run_command", "", "out-1"},
		[3]string{"run_command", "", "out-2"},
		[3]string{"run_command", "", "out-3"},
	)
	got := results(apply(t, DefaultConfig(), msgs))
	if !strings.HasPrefix(got[0], "[tool result evicted:") || !strings.HasPrefix(got[1], "[tool result evicted:") {
		t.Fatalf("unkeyed read + oldest cmd must age out, got %q, %q", got[0], got[1])
	}
}

func TestAttachDetach(t *testing.T) {
	h := bridle.NewHarness(nil)
	detach := Attach(h, DefaultConfig())
	detach() // must not panic; hook must unregister cleanly
}
