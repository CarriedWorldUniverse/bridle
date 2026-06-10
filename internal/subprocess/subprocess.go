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
	"strings"
	"time"
)

// graceWindow is how long WatchCancel waits after the graceful
// termination signal before escalating to SIGKILL.
const graceWindow = 5 * time.Second

// MergeEnv overlays the per-turn key=value map onto the parent
// process's env. Per-turn keys take precedence; any KEY=VALUE pair in
// `base` whose KEY is in `overlay` is replaced. Both inputs are left
// unmodified.
func MergeEnv(base []string, overlay map[string]string) []string {
	if len(overlay) == 0 {
		return base
	}
	// Index base by KEY for O(1) replacement.
	idx := make(map[string]int, len(base))
	out := make([]string, len(base))
	copy(out, base)
	for i, kv := range out {
		if eq := strings.IndexByte(kv, '='); eq > 0 {
			idx[kv[:eq]] = i
		}
	}
	for k, v := range overlay {
		entry := k + "=" + v
		if i, ok := idx[k]; ok {
			out[i] = entry
		} else {
			out = append(out, entry)
		}
	}
	return out
}

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
