package distill

import (
	"errors"
	"strings"
	"testing"
)

type fakeSum struct {
	err   error
	focus string
}

func (f *fakeSum) Distill(text, focus string) (string, error) {
	f.focus = focus
	if f.err != nil {
		return "", f.err
	}
	return "SUMMARY(" + text[:10] + "...)", nil
}

func big(n int) string { return strings.Repeat("x", n) }

func TestSmallResultPassesThrough(t *testing.T) {
	d := New(&fakeSum{}, 100)
	out := d.Process("Read", "short output", "task")
	if out != "short output" {
		t.Fatalf("small result must pass through unchanged, got %q", out)
	}
}

func TestLargeResultDistilledWithEscalation(t *testing.T) {
	fs := &fakeSum{}
	d := New(fs, 100)
	raw := big(500)
	out := d.Process("Read", raw, "find the signature")
	if !strings.Contains(out, "SUMMARY(") {
		t.Fatalf("large result must be distilled, got %q", out)
	}
	if !strings.Contains(out, "read_raw") || !strings.Contains(out, "handle=") {
		t.Fatalf("distilled result must advertise the escalation path, got %q", out)
	}
	if fs.focus != "find the signature" {
		t.Fatalf("distillation must receive the task focus, got %q", fs.focus)
	}
	// extract the handle and confirm the raw is retrievable verbatim
	i := strings.Index(out, `handle="`)
	h := out[i+8:]
	h = h[:strings.Index(h, `"`)]
	got, ok := d.Raw(h)
	if !ok || got != raw {
		t.Fatalf("verbatim raw must be retrievable by handle (ok=%v, match=%v)", ok, got == raw)
	}
}

func TestFailOpenOnSummariserError(t *testing.T) {
	d := New(&fakeSum{err: errors.New("model down")}, 100)
	raw := big(500)
	out := d.Process("Bash", raw, "task")
	if out != raw {
		t.Fatal("summariser error must FAIL OPEN — return raw, never drop data")
	}
}

func TestNilDistillerIsPassThrough(t *testing.T) {
	var d *Distiller
	out := d.Process("Read", big(9000), "task")
	if out != big(9000) {
		t.Fatal("nil distiller must pass through (disabled)")
	}
}
