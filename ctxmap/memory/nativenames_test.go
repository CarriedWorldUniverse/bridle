package memory

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/CarriedWorldUniverse/bridle/ctxmap/render"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/store"
)

func wsEngine(t *testing.T) *Engine {
	t.Helper()
	st, _ := store.Open(":memory:")
	t.Cleanup(func() { st.Close() })
	rend, _ := render.New(st)
	e := New(Config{SessionID: "native"}, st, rend, nil, nil, nil)
	t.Cleanup(func() { e.Close() })
	e.EnableWorkingState()
	return e
}

// The fs tools advertise Claude's native names and a file_path argument.
// Working-state tracking must follow, or a session's whole "files
// created/edited" list silently reads "(none yet)" while it edits.
func TestWorkingState_NativeNamesAndFilePathArg(t *testing.T) {
	e := wsEngine(t)
	e.ObserveTool("Read", json.RawMessage(`{"file_path":"src/framing.py"}`), `"..."`)
	e.ObserveTool("Write", json.RawMessage(`{"file_path":"src/const.py","content":"x"}`), `"ok"`)
	e.ObserveTool("Edit", json.RawMessage(`{"file_path":"src/const.py","old_string":"a","new_string":"b"}`), `"ok"`)

	b := e.WorkingMemoryBlock("task")
	for _, want := range []string{"src/const.py (2×)", "Read src/framing.py", "Write src/const.py"} {
		if !strings.Contains(b, want) {
			t.Fatalf("working-state block missing %q\n---\n%s", want, b)
		}
	}
	if strings.Contains(b, "(none yet)") {
		t.Error("files created/edited is empty despite two writes — the native names were not tracked")
	}
}

// A native NAME paired with the legacy ARG is a real combination: the fs
// family accepts either spelling, so tracking must too.
func TestWorkingState_NativeNameWithLegacyArg(t *testing.T) {
	e := wsEngine(t)
	e.ObserveTool("Write", json.RawMessage(`{"path":"mixed.py","content":"x"}`), `"ok"`)
	if b := e.WorkingMemoryBlock("task"); !strings.Contains(b, "mixed.py") {
		t.Fatalf("a native name with the legacy path arg was dropped\n---\n%s", b)
	}
}

// The legacy spellings must keep working — a resumed pre-rename thread
// replays them.
func TestWorkingState_LegacyNamesStillTracked(t *testing.T) {
	e := wsEngine(t)
	e.ObserveTool("write_file", json.RawMessage(`{"path":"old.py","content":"x"}`), `"ok"`)
	e.ObserveTool("read_file", json.RawMessage(`{"path":"old2.py"}`), `"..."`)

	b := e.WorkingMemoryBlock("task")
	if !strings.Contains(b, "old.py") || !strings.Contains(b, "read_file old2.py") {
		t.Fatalf("legacy tool names stopped being tracked\n---\n%s", b)
	}
}
