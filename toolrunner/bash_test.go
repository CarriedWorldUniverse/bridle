package toolrunner

import (
	"context"
	"encoding/json"
	"testing"
)

func newTestRunner(t *testing.T) *LocalToolRunner {
	t.Helper()
	r, err := New(Config{WorkDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestBashEcho(t *testing.T) {
	r := newTestRunner(t)
	out, err := r.runBash(context.Background(), json.RawMessage(`{"command":"echo hello"}`))
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	var res bashResult
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 0 || res.Stdout != "hello\n" {
		t.Fatalf("got %+v", res)
	}
}

func TestBashNonZeroIsToolResultNotError(t *testing.T) {
	r := newTestRunner(t)
	out, err := r.runBash(context.Background(), json.RawMessage(`{"command":"exit 3"}`))
	if err != nil {
		t.Fatalf("non-zero exit must be a tool result, not a Go error: %v", err)
	}
	var res bashResult
	_ = json.Unmarshal(out, &res)
	if res.ExitCode != 3 {
		t.Fatalf("want exit 3, got %+v", res)
	}
}
