// Package subprocess holds plumbing shared by the subprocess-stream
// providers (claudecode, codexcli, geminicli): the cancel watcher,
// env merging, prompt extraction, JSONL stream scanning, and stderr
// error classification. The helpers are mechanical extractions of
// previously duplicated provider code; provider-specific behavior
// (event schemas, error-pattern tables, signal choice) stays in each
// provider.
package subprocess

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// scanBufBytes is the line buffer cap for ScanJSONLines. CLI stream
// events can carry large embedded payloads (full tool results, spilled
// file contents), so the default 64K bufio limit is not enough.
const scanBufBytes = 1024 * 1024

// ScanJSONLines reads r line by line with a 1MB buffer, trims
// whitespace, skips empty lines, and hands every remaining line to
// perLine. Event-schema dispatch — including skipping non-JSON banner
// lines a CLI may print on stdout — stays inside each provider's
// callback.
//
// The line slice aliases the scanner's internal buffer and is only
// valid for the duration of the callback.
//
// Returns the scanner error, if any (io.EOF is treated as clean
// termination); callers wrap it with their provider prefix.
func ScanJSONLines(r io.Reader, perLine func(line []byte)) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, scanBufBytes), scanBufBytes)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		perLine(line)
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return err
	}
	return nil
}

// LastUserPrompt returns the most recent user message — NOT the full
// SessionTail — for use as a CLI prompt argument.
//
// Subprocess-stream CLIs get prior conversation history from their own
// session store on resume, not from argv. Folding SessionTail into the
// prompt would (a) duplicate the history the subprocess is already
// loading from disk and (b) blow Windows CreateProcess's 32K argv
// budget once a session accumulates state. Observed 2026-05-13: keel
// as Frame (global context) crossed 32K after a few turns and every
// spawn failed with the misleading "filename or extension is too long"
// kernel error. See task #216 for full diagnosis.
//
// Direct-API providers need history reassembled because they have no
// subprocess-owned session store — they must not use this function.
func LastUserPrompt(msgs []bridle.ProviderMessage) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return msgs[i].Content
		}
	}
	return ""
}

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
