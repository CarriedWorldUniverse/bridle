package openai

import (
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func TestSanitizeToolName_ValidNamesUnchanged(t *testing.T) {
	for _, n := range []string{"read_file", "run_command", "web_fetch", "agent", "mcp__github__search"} {
		if got := sanitizeToolName(n); got != n {
			t.Errorf("sanitizeToolName(%q) = %q; an already-valid name must be unchanged", n, got)
		}
	}
}

// The exact bug this whole unit exists to fix: agora's dotted tool names
// (task.write, memory.read, bg.output) pass fine on Anthropic's more
// permissive API and only 400 here — verified live against the real API
// before this fix existed (400: "function name is invalid, must start
// with a letter and can contain letters, numbers, underscores, and
// dashes").
func TestSanitizeToolName_DotsBecomeUnderscores(t *testing.T) {
	cases := map[string]string{
		"task.write":  "task_write",
		"memory.read": "memory_read",
		"bg.output":   "bg_output",
		"a.b.c":       "a_b_c",
	}
	for in, want := range cases {
		if got := sanitizeToolName(in); got != want {
			t.Errorf("sanitizeToolName(%q) = %q; want %q", in, got, want)
		}
	}
}

// MCP tool names come from a THIRD-PARTY SERVER, entirely outside any
// bridle caller's control — a server can name a tool anything, including
// something starting with a digit or containing spaces/slashes.
func TestSanitizeToolName_ArbitraryMCPServerNames(t *testing.T) {
	cases := map[string]bool{ // value unused; just must not panic and must satisfy the charset
		"1-lookup":   true,
		"search web": true,
		"a/b":        true,
		"":           true,
		"日本語":        true,
	}
	for in := range cases {
		got := sanitizeToolName(in)
		if !toolNameRe.MatchString(got) {
			t.Errorf("sanitizeToolName(%q) = %q; still fails the charset", in, got)
		}
	}
}

func TestSanitizeToolName_IsPure(t *testing.T) {
	// The property the whole design leans on: called independently, twice,
	// with no shared state, it must agree with itself — this is what lets
	// toOpenAIMessages (history replay) and toOpenAITools (current turn's
	// listing) derive the same wire name without coordinating.
	for _, n := range []string{"task.write", "bg.output", "1-lookup", "read_file"} {
		a := sanitizeToolName(n)
		b := sanitizeToolName(n)
		if a != b {
			t.Fatalf("sanitizeToolName(%q) is not pure: %q vs %q", n, a, b)
		}
	}
}

func TestToOpenAITools_SanitizesAndErrorsOnCollision(t *testing.T) {
	_, err := toOpenAITools([]bridle.ToolDef{
		{Name: "task.write"},
		{Name: "task_write"}, // distinct name, SAME sanitized wire form
	})
	if err == nil {
		t.Fatal("two names colliding after sanitization were accepted silently")
	}
}

func TestToOpenAITools_NoCollisionForDistinctNames(t *testing.T) {
	tools, err := toOpenAITools([]bridle.ToolDef{
		{Name: "task.write"},
		{Name: "task.read"},
		{Name: "bg.output"},
	})
	if err != nil {
		t.Fatalf("toOpenAITools: %v", err)
	}
	got := map[string]bool{}
	for _, tl := range tools {
		got[tl.Function.Name] = true
	}
	for _, want := range []string{"task_write", "task_read", "bg_output"} {
		if !got[want] {
			t.Errorf("wire tools missing %q; got %v", want, got)
		}
	}
}

func TestUnsanitizeToolName_ReversesAgainstTheActualDefs(t *testing.T) {
	defs := []bridle.ToolDef{{Name: "task.write"}, {Name: "read_file"}}
	if got := unsanitizeToolName("task_write", defs); got != "task.write" {
		t.Errorf("unsanitizeToolName(task_write) = %q; want task.write", got)
	}
	// Already-valid names round-trip as themselves.
	if got := unsanitizeToolName("read_file", defs); got != "read_file" {
		t.Errorf("unsanitizeToolName(read_file) = %q; want read_file", got)
	}
}

// A wire name outside the sent set (should not happen, but must not
// panic or silently invent a wrong mapping) passes through unchanged.
func TestUnsanitizeToolName_UnknownNamePassesThrough(t *testing.T) {
	defs := []bridle.ToolDef{{Name: "task.write"}}
	if got := unsanitizeToolName("something_else", defs); got != "something_else" {
		t.Errorf("unsanitizeToolName(unknown) = %q; want passthrough", got)
	}
}
