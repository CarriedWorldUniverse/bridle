package codemap

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func writeRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "src"), 0o755)
	os.WriteFile(filepath.Join(dir, "src", "crc.py"), []byte(`import zlib

def crc32(data: bytes) -> int:
    """IEEE CRC-32 of data as an unsigned 32-bit integer."""
    return zlib.crc32(data) & 0xFFFFFFFF
`), 0o644)
	os.WriteFile(filepath.Join(dir, "src", "framing.py"), []byte(`import struct
from .crc import crc32

def encode_frame(type_code: int,
                 payload: bytes) -> bytes:
    """Wrap a payload in a frame."""
    return struct.pack(">I", crc32(payload))

class Reader:
    def decode(self, buf):
        return crc32(buf)
`), 0o644)
	return dir
}

func needPy(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not on PATH")
	}
}

func TestIndexAndQueries(t *testing.T) {
	needPy(t)
	e := New(writeRepo(t), PyIndexer{})

	out := e.Outline("src/framing.py")
	for _, want := range []string{"encode_frame", "class Reader", "method def decode", "from .crc import crc32"} {
		if !strings.Contains(out, want) {
			t.Fatalf("outline missing %q:\n%s", want, out)
		}
	}
	// multi-line signature captured whole
	if !strings.Contains(out, "def encode_frame(type_code: int, payload: bytes) -> bytes") {
		t.Fatalf("multi-line signature not joined:\n%s", out)
	}
	if s := e.Symbol("crc32"); !strings.Contains(s, "src/crc.py:3") || !strings.Contains(s, "IEEE CRC-32") {
		t.Fatalf("symbol lookup wrong:\n%s", s)
	}
	if r := e.Refs("crc32"); !strings.Contains(r, "src/framing.py") {
		t.Fatalf("refs missing caller:\n%s", r)
	}
	if b := e.Body("crc32"); !strings.Contains(b, "return zlib.crc32(data)") {
		t.Fatalf("body span wrong:\n%s", b)
	}
	if d := e.Diag(); d != "no inconsistencies found" {
		t.Fatalf("clean repo should have no diagnostics:\n%s", d)
	}
}

func TestDriftOnWrite(t *testing.T) {
	needPy(t)
	dir := writeRepo(t)
	e := New(dir, PyIndexer{})

	// rename crc32 -> compute_crc: framing.py's import + calls now dangle
	os.WriteFile(filepath.Join(dir, "src", "crc.py"), []byte(`import zlib

def compute_crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF
`), 0o644)
	drift := e.Reindex("src/crc.py")
	joined := strings.Join(drift, "\n")
	if !strings.Contains(joined, "crc32 REMOVED") {
		t.Fatalf("drift must flag the removed symbol:\n%s", joined)
	}
	if !strings.Contains(joined, "src/framing.py") {
		t.Fatalf("drift must name the referencing file (blast radius):\n%s", joined)
	}
	// and diagnostics now see the broken import
	if d := e.Diag(); !strings.Contains(d, `import "crc32"`) {
		t.Fatalf("diag must flag the dangling import:\n%s", d)
	}
}

func TestSyntaxErrorSurfaces(t *testing.T) {
	needPy(t)
	dir := writeRepo(t)
	e := New(dir, PyIndexer{})
	os.WriteFile(filepath.Join(dir, "src", "crc.py"), []byte("def broken(:\n    pass\n"), 0o644)
	drift := strings.Join(e.Reindex("src/crc.py"), "\n")
	if !strings.Contains(drift, "syntax error") {
		t.Fatalf("drift must surface the parse error:\n%s", drift)
	}
}

func TestSweepChanged(t *testing.T) {
	needPy(t)
	dir := writeRepo(t)
	e := New(dir, PyIndexer{})
	// external mutation (as a run_command would do)
	os.WriteFile(filepath.Join(dir, "src", "crc.py"), []byte("def other():\n    pass\n"), 0o644)
	drift := strings.Join(e.SweepChanged(), "\n")
	if !strings.Contains(drift, "crc32 REMOVED") {
		t.Fatalf("sweep must catch external file changes:\n%s", drift)
	}
}

func TestFailOpenWithoutIndex(t *testing.T) {
	e := New(t.TempDir(), failingIndexer{})
	if s := e.Symbol("anything"); s != unavailable {
		t.Fatalf("dead indexer must degrade to unavailable, got %q", s)
	}
	if d := e.Reindex("nope.py"); d != nil {
		t.Fatalf("reindex with dead indexer must be silent, got %v", d)
	}
}

type failingIndexer struct{}

func (failingIndexer) Index(string, []string) (map[string]*FileIndex, error) {
	return nil, os.ErrNotExist
}
