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

// Pattern maps a set of case-insensitive substring patterns to a
// bridle.ProviderError classification. Order matters: Classify
// iterates a pattern list top-to-bottom and returns the first match,
// so each provider orders its own list (e.g. auth before the generic
// "exit" path, network before timeout where the network signal is
// more actionable).
//
// Patterns are matched against the CLI's stderr (lowercased). They're
// mostly stable API error codes ("authentication_failed",
// "rate_limit_error", "overloaded_error") that surface either as plain
// text or embedded in stream-json error events; a few are network
// shell errors ("connection refused", etc.).
type Pattern struct {
	Kind     bridle.ProviderErrorKind
	Patterns []string
	Message  string
}

// Classify matches the CLI's stderr against the provider's ordered
// pattern list and returns the first matching classification. ok is
// false when nothing matches; callers supply their own fallback
// (typically a generic subprocess-exit ProviderError).
func Classify(stderr string, patterns []Pattern) (kind bridle.ProviderErrorKind, msg string, ok bool) {
	lower := strings.ToLower(stderr)
	for _, p := range patterns {
		for _, sub := range p.Patterns {
			if strings.Contains(lower, sub) {
				return p.Kind, p.Message, true
			}
		}
	}
	return "", "", false
}

// SharedPatterns returns the cross-provider, actionable error-class
// table used as a fallback layer beneath each CLI provider's own
// pattern list. It recognizes the failure classes that bite operators
// in headless/dispatch lanes regardless of which CLI surfaced them,
// and maps each to a ProviderError kind with a message that says WHAT
// to do.
//
// The set is intentionally extendable: a provider composes its own
// (more specific, CLI-worded) patterns FIRST and appends these so a
// generic signal still classifies. Order within this list is
// significant — auth before rate before network before config before
// crash — and Classify returns the first match, so the more
// actionable / more specific classes lead.
//
// Each provider passes its own ID into the messages via the prefix
// argument so the surfaced text names the failing CLI ("codexcli:
// provider auth failed …").
//
// The substrings are deliberately broad, code-level signals:
//   - AUTH: HTTP 401/403, "unauthorized", "forbidden", "invalid api
//     key", "expired"/"token expired", "login required". This is the
//     class that made a codex 401 (expired auth.json / wrong endpoint)
//     surface as a bare exit-1 before NEX-588 — "401"/"unauthorized"
//     now classify as AUTH even when the CLI prints no friendly
//     "not logged in" string.
//   - RATE_LIMIT: HTTP 429, "rate limit"/"rate_limit", "quota",
//     "too many requests".
//   - NETWORK: connection refused/reset, DNS failures ("no such
//     host", "name resolution"), "network is unreachable", "timeout"
//     at the socket layer.
//   - CONFIG: binary not found, missing config/profile, "no such
//     file", "command not found" — non-transient setup faults.
//   - SUBPROCESS_CRASH: fatal signals (segfault/SIGSEGV/SIGABRT),
//     OOM ("out of memory"/"killed"), "panic"/"stack overflow".
func SharedPatterns(prefix string) []Pattern {
	return []Pattern{
		{
			Kind: bridle.ProviderErrorAuthFailed,
			Patterns: []string{
				"401", "403", "unauthorized", "forbidden",
				"invalid api key", "invalid_api_key", "invalid x-api-key",
				"authentication", "not logged in", "login required",
				"login-required", "expired", "token expired",
				"permission denied (publickey", // git/ssh auth, distinct from filesystem perms
			},
			Message: prefix + ": provider auth failed (401/403) — refresh credentials (check auth.json / API key / base URL)",
		},
		{
			Kind: bridle.ProviderErrorRateLimit,
			Patterns: []string{
				"429", "rate limit", "rate_limit", "ratelimit",
				"quota", "too many requests",
			},
			Message: prefix + ": rate limited (429/quota) — back off and retry later",
		},
		{
			Kind: bridle.ProviderErrorNetworkError,
			Patterns: []string{
				"connection refused", "connection reset", "no route to host",
				"network is unreachable", "no such host", "name resolution",
				"dns", "could not resolve", "i/o timeout", "tcp",
			},
			Message: prefix + ": network error reaching provider — check connectivity / DNS / proxy",
		},
		{
			Kind: bridle.ProviderErrorConfig,
			Patterns: []string{
				"command not found", "executable file not found",
				"no such file or directory", "not found in $path",
				"missing config", "config not found", "unknown profile",
				"unknown flag", "invalid argument",
			},
			Message: prefix + ": configuration error — missing binary/config (not retryable; fix setup)",
		},
		{
			Kind: bridle.ProviderErrorCrash,
			Patterns: []string{
				"segmentation fault", "sigsegv", "sigabrt", "core dumped",
				"out of memory", "oomkilled", "killed", "fatal error",
				"stack overflow", "runtime: out of memory",
			},
			Message: prefix + ": subprocess crashed (signal/OOM) — the CLI process itself died abnormally",
		},
	}
}

// ClassifyWithFallback matches stderr against the provider's own
// pattern list first, then the shared cross-provider table, returning
// the first match from either. This is the recommended classification
// entry point for subprocess providers: provider-specific patterns win
// (they carry CLI-accurate wording) while the shared layer guarantees a
// bare "401" / "429" / "segfault" still gets an actionable kind instead
// of falling through to a generic exit-status error.
//
// prefix is the provider ID used in the shared messages.
func ClassifyWithFallback(stderr, prefix string, providerPatterns []Pattern) (kind bridle.ProviderErrorKind, msg string, ok bool) {
	if k, m, found := Classify(stderr, providerPatterns); found {
		return k, m, true
	}
	return Classify(stderr, SharedPatterns(prefix))
}

// resumeNotFoundSignals are the stderr substrings (lowercased) that
// indicate a session resume failed because the referenced session is
// genuinely missing or corrupt — NOT because of a transient/auth/
// network fault. This distinction matters: a missing session should
// degrade to a fresh session (the prior context is unrecoverable
// anyway), whereas a transient error must be retried against the SAME
// session so a builder mid-task does not silently lose its context.
//
// The list is the cross-provider union of how the CLIs word a missing
// session: codex ("no such thread", "session not found", "thread … not
// found"), claude-code ("no conversation found", "session … not
// found"/"does not exist"), gemini ("no checkpoint", "invalid resume
// index"). Extend here when a new provider/CLI version surfaces a
// different phrasing.
var resumeNotFoundSignals = []string{
	"session not found",
	"no such session",
	"no such thread",
	"thread not found",
	"no conversation found",
	"conversation not found",
	"does not exist",
	"no checkpoint",
	"invalid resume",
	"corrupt", // corrupt session file / corrupt checkpoint
	"unknown session",
	"unknown thread",
}

// IsResumeNotFound reports whether stderr indicates a resume failed
// because the session is missing/corrupt (vs a transient or auth
// error). Callers use it to decide whether to degrade to a fresh
// session (true) or surface/retry the error as-is (false). Matched
// case-insensitively as a substring.
func IsResumeNotFound(stderr string) bool {
	lower := strings.ToLower(stderr)
	for _, sig := range resumeNotFoundSignals {
		if strings.Contains(lower, sig) {
			return true
		}
	}
	return false
}

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
//
// "Most recent user message" means the whole TRAILING RUN of user
// messages, joined in order — not the literal last element. A harness
// layer may append its own user-role message after the operator's (the
// ctxmap adapter appends a working-memory block last, for direct-API
// prompt-cache stability); taking only the literal last message made
// that block SHADOW the operator's text, and every subprocess turn ran
// with no task while the model politely stood by (observed live
// 2026-07-21, fable via claudesdk). Prior turns' messages are still
// excluded — the argv-budget rationale above is about history, and a
// trailing run is one turn's worth.
func LastUserPrompt(msgs []bridle.ProviderMessage) string {
	end := len(msgs)
	for end > 0 && msgs[end-1].Role != "user" {
		end--
	}
	start := end
	for start > 0 && msgs[start-1].Role == "user" {
		start--
	}
	parts := make([]string, 0, end-start)
	for _, m := range msgs[start:end] {
		parts = append(parts, m.Content)
	}
	return strings.Join(parts, "\n\n")
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

// WatchCancelGroup is WatchCancel's opt-in PROCESS-GROUP variant: on ctx
// cancellation it signals the whole process group led by cmd (via
// signalGroup/killGroup -- see procgroup_unix.go/procgroup_windows.go),
// instead of only cmd.Process, so a grandchild the direct child spawned
// (e.g. claudesdk's sidecar spawning the real `claude` CLI) is reaped
// too, not left orphaned.
//
// Callers MUST have called SetPgid(cmd) before cmd.Start() for this to
// have group semantics on unix; without it, signalGroup/killGroup
// degrade to signaling only cmd.Process (same as WatchCancel).
//
// This is a SEPARATE function, not a change to WatchCancel's behavior --
// claudecode/codexcli/geminicli keep calling WatchCancel unchanged and
// are completely unaffected by this addition (NEX-745 review gate: the
// process-group kill is opt-in per spawn call, not a global change to
// the shared cancel-watcher contract).
func WatchCancelGroup(ctx context.Context, cmd *exec.Cmd, procExited <-chan struct{}, termSignal os.Signal) {
	select {
	case <-ctx.Done():
		_ = signalGroup(cmd, termSignal)
		timer := time.NewTimer(graceWindow)
		defer timer.Stop()
		select {
		case <-timer.C:
			_ = killGroup(cmd)
		case <-procExited:
			// Process (group) exited cleanly during grace period -- no
			// SIGKILL needed.
		}
	case <-procExited:
		// Natural exit -- nothing to do.
	}
}
