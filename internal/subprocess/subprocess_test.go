package subprocess

import (
	"context"
	"os/exec"
	"testing"
	"time"
)

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
