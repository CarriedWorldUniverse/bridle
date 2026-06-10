package subprocess

import (
	"context"
	"os/exec"
	"reflect"
	"sort"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func TestLastUserPrompt(t *testing.T) {
	cases := []struct {
		name string
		msgs []bridle.ProviderMessage
		want string
	}{
		{"nil messages", nil, ""},
		{"no user messages", []bridle.ProviderMessage{
			{Role: "system", Content: "sys"},
			{Role: "assistant", Content: "hi"},
		}, ""},
		{"single user", []bridle.ProviderMessage{
			{Role: "user", Content: "hello"},
		}, "hello"},
		{"most recent user wins", []bridle.ProviderMessage{
			{Role: "user", Content: "first"},
			{Role: "assistant", Content: "reply"},
			{Role: "user", Content: "second"},
			{Role: "assistant", Content: "trailing"},
		}, "second"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := LastUserPrompt(tc.msgs); got != tc.want {
				t.Errorf("LastUserPrompt = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestMergeEnv(t *testing.T) {
	base := []string{"PATH=/usr/bin", "HOME=/home/x", "FOO=old"}

	t.Run("empty overlay returns base unchanged", func(t *testing.T) {
		got := MergeEnv(base, nil)
		if !reflect.DeepEqual(got, base) {
			t.Errorf("MergeEnv(base, nil) = %v, want %v", got, base)
		}
	})

	t.Run("overlay replaces and appends", func(t *testing.T) {
		got := MergeEnv(base, map[string]string{"FOO": "new", "BAR": "added"})
		sorted := append([]string(nil), got...)
		sort.Strings(sorted)
		want := []string{"BAR=added", "FOO=new", "HOME=/home/x", "PATH=/usr/bin"}
		if !reflect.DeepEqual(sorted, want) {
			t.Errorf("MergeEnv = %v, want (any order) %v", got, want)
		}
		// Replacement must happen in place, not append a duplicate.
		if len(got) != 4 {
			t.Errorf("len = %d, want 4 (no duplicate FOO)", len(got))
		}
	})

	t.Run("inputs are left unmodified", func(t *testing.T) {
		baseCopy := append([]string(nil), base...)
		_ = MergeEnv(base, map[string]string{"FOO": "new"})
		if !reflect.DeepEqual(base, baseCopy) {
			t.Errorf("base mutated: %v", base)
		}
	})
}

// TestWatchCancelSignalsOnCancel: cancelling the context must terminate
// a long-running child via the graceful signal well inside the grace
// window (sleep exits on SIGTERM, so no SIGKILL escalation is needed).
func TestWatchCancelSignalsOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}

	procExited := make(chan struct{})
	watcherDone := make(chan struct{})
	go func() {
		defer close(watcherDone)
		WatchCancel(ctx, cmd, procExited, TermSignal())
	}()

	cancel()

	waitDone := make(chan error, 1)
	go func() { waitDone <- cmd.Wait() }()

	select {
	case err := <-waitDone:
		if err == nil {
			t.Fatalf("expected sleep to be killed by signal, got clean exit")
		}
	case <-time.After(3 * time.Second):
		_ = cmd.Process.Kill()
		t.Fatalf("process not terminated within 3s of cancel")
	}
	close(procExited)

	select {
	case <-watcherDone:
	case <-time.After(2 * time.Second):
		t.Fatalf("watcher did not return after procExited closed")
	}
}

// TestWatchCancelNaturalExit: when the process exits on its own and
// procExited is closed, the watcher must return without signaling.
func TestWatchCancelNaturalExit(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := exec.Command("true")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}

	procExited := make(chan struct{})
	watcherDone := make(chan struct{})
	go func() {
		defer close(watcherDone)
		WatchCancel(ctx, cmd, procExited, TermSignal())
	}()

	if err := cmd.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	close(procExited)

	select {
	case <-watcherDone:
	case <-time.After(2 * time.Second):
		t.Fatalf("watcher did not return after natural exit")
	}
}
