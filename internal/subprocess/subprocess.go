// Package subprocess holds plumbing shared by the subprocess-stream
// providers (claudecode, codexcli, geminicli): the cancel watcher,
// env merging, prompt extraction, JSONL stream scanning, and stderr
// error classification. The helpers are mechanical extractions of
// previously duplicated provider code; provider-specific behavior
// (event schemas, error-pattern tables, signal choice) stays in each
// provider.
package subprocess

import (
	"context"
	"os"
	"os/exec"
	"time"
)

// graceWindow is how long WatchCancel waits after the graceful
// termination signal before escalating to SIGKILL.
const graceWindow = 5 * time.Second

// WatchCancel implements the cancel watcher for a spawned CLI process:
// on ctx cancellation it sends termSignal, waits up to a 5s grace
// period for procExited, then Kills the process.
//
// procExited must be closed AFTER cmd.Wait() returns. The watcher
// waits on either ctx cancellation OR the process exiting naturally;
// on cancellation it sends termSignal and waits up to the grace
// period for procExited before SIGKILLing. Without procExited being
// closed externally, the watcher would (a) leak on natural exit and
// (b) always SIGKILL after the full grace period even when the
// process already responded to the termination signal.
//
// WatchCancel blocks until the watcher resolves; callers run it in a
// goroutine alongside cmd.Wait().
func WatchCancel(ctx context.Context, cmd *exec.Cmd, procExited <-chan struct{}, termSignal os.Signal) {
	select {
	case <-ctx.Done():
		_ = cmd.Process.Signal(termSignal)
		timer := time.NewTimer(graceWindow)
		defer timer.Stop()
		select {
		case <-timer.C:
			_ = cmd.Process.Kill()
		case <-procExited:
			// Process exited cleanly during grace period — no SIGKILL needed.
		}
	case <-procExited:
		// Natural exit — nothing to do.
	}
}
