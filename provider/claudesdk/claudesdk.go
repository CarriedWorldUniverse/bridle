// Package claudesdk implements the bridle Provider interface via a thin
// TypeScript sidecar (bridle-claude-sidecar) that wraps the official
// @anthropic-ai/claude-agent-sdk over stdio JSON-lines.
//
// Category: subprocess-stream, like claudecode — but UNLIKE claudecode
// this lane supports custom tools (Capabilities().SupportsCustomTools =
// true): Claude's tool round-trip happens inside the sidecar's live
// SDK-MCP server, and each custom tool call the sidecar reports is
// serviced by bridle's own ToolRunner-backed pipeline via
// ProviderRequest.ToolExecutor (bridle spec NEX-745 §4) BEFORE the
// sidecar's query() is allowed to continue. Because that whole
// round-trip completes inside one RunTurn call, the ProviderResult this
// package returns always carries an EMPTY ToolCalls slice — see run.go's
// comment at the "NEX-745" tag for why that's load-bearing (it's what
// keeps the harness's per-round tool-execution loop from re-running the
// same calls a second time).
//
// v1 is per-turn spawn + resume, mirroring claudecode's session model
// (bridle-agentsdk-spec.md §5,§9): the sidecar is spawned fresh for
// every turn; continuity is carried via the init line's session id
// (SessionID for a fresh caller-chosen id, Resume for continuing one),
// same split as claudecode's --session-id/--resume.
//
// Auth: the token is only ever ENVIRONMENT for the SDK
// (CLAUDE_CODE_OAUTH_TOKEN via ProviderEnv, or ambient
// ~/.claude/.credentials.json) — this package never reads it to
// construct its own HTTP request (spec §1, §6 — the banned CLIProxyAPI
// pattern). The sidecar's spawn env is also scrubbed of
// ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN (see scrubAuthEnv) so an
// ambient metered API key set for a co-located claude-api lane can't
// silently outrank CLAUDE_CODE_OAUTH_TOKEN in the vendored SDK's auth
// precedence.
package claudesdk

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/internal/subprocess"
)

const providerID = bridle.ProviderClaudeSDK

// protocolVersion is the init line's version field. Bump when the wire
// schema (wire.go) changes in a way an older-built sidecar can't parse;
// the sidecar is expected to reject a mismatch loudly at turn start
// (spec §5,§7 — "never mid-turn").
const protocolVersion = 1

// Mode selects which native (built-in Claude Code) tools the sidecar
// exposes alongside bridle's custom tools (spec §3).
type Mode string

const (
	// ModeFunnel is the default: ALL native tools disabled, Claude sees
	// only ProviderRequest.Tools. Pure model-funnel (agora-spec-bridle
	// §4) — this is what the OpenAI-compatible front and aspect-with-
	// bridle-tools callers want.
	ModeFunnel Mode = "funnel"
	// ModeAgent enables the native toolset per AllowedTools/
	// DisallowedTools — parity with today's claudecode, for aspect
	// seats that want the full harness.
	ModeAgent Mode = "agent"
)

// Provider implements bridle.Provider by spawning bridle-claude-sidecar.
// Config mirrors claudecode.Provider where the semantics carry over.
type Provider struct {
	// SidecarPath is the path to the bridle-claude-sidecar entry point
	// (the built dist/index.js, run via `node`, or a wrapper script).
	// Defaults to "bridle-claude-sidecar" (PATH lookup) via New().
	SidecarPath string

	// Mode selects native-tool exposure. Zero value is ModeFunnel — the
	// safer default (funnel-mode callers must opt IN to native tools,
	// not opt out of them).
	Mode Mode

	// AllowedTools/DisallowedTools restrict the CLI-native toolset in
	// ModeAgent. Ignored in ModeFunnel (native tools are all off there
	// regardless).
	AllowedTools    []string
	DisallowedTools []string

	// ExtraOpts is passed through to the sidecar's SDK `options` object
	// verbatim, for options this provider doesn't have a named
	// field/knob for yet. Each value must be valid JSON.
	ExtraOpts map[string]json.RawMessage

	// MaxRetries is the maximum number of retry attempts for transient
	// errors (rate_limit, server_error, network_error, timeout). 0 = no
	// retry. Mirrors claudecode.Provider.MaxRetries.
	MaxRetries int
	// RetryDelay is the initial delay between retries (doubles each
	// attempt). Default: 2s when MaxRetries > 0.
	RetryDelay time.Duration
}

// New returns a claudesdk Provider with default settings: PATH-resolved
// sidecar, funnel mode.
func New() *Provider {
	return &Provider{SidecarPath: "bridle-claude-sidecar", Mode: ModeFunnel}
}

func (p *Provider) Name() bridle.ProviderID { return providerID }

func (p *Provider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{
		Category:               bridle.CategorySubprocessStream,
		SupportsCustomTools:    true, // THE point of this lane
		SupportsBeforeToolCall: true, // via SDK canUseTool
		SupportsAfterToolCall:  true, // custom calls: real hook fire via ToolExecutor/executeToolCall; native_tool observations: sink-only (see below)
		SupportsMCP:            false,
	}
}

// RunTurn spawns the sidecar, sends one init line, pumps its normalized
// event stream to sink, services custom tool calls via
// req.ToolExecutor, and returns once the sidecar reports "done" (or the
// process exits/errors). Cancellation via ctx sends SIGTERM then SIGKILL
// after a grace period — to the WHOLE process group the sidecar leads
// (internal/subprocess.SetPgid + WatchCancelGroup), not just the
// sidecar's own PID, because the sidecar spawns the real `claude` CLI as
// its own child and a single-PID signal would leave it orphaned. This
// differs from claudecode/codexcli/geminicli, which use plain
// WatchCancel unchanged (they have no such grandchild to reap).
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	maxRetries := p.MaxRetries
	retryDelay := p.RetryDelay
	if retryDelay == 0 && maxRetries > 0 {
		retryDelay = 2 * time.Second
	}

	for attempt := 0; ; attempt++ {
		if ctx.Err() != nil {
			return bridle.ProviderResult{}, ctx.Err()
		}

		result, err := p.runTurnOnce(ctx, req, sink)
		if err == nil {
			return result, nil
		}
		if !isRetryable(err) {
			return result, err
		}
		if attempt >= maxRetries {
			return result, err
		}
		delay := retryDelay * (1 << attempt)
		if delay > 30*time.Second {
			delay = 30 * time.Second
		}
		sink.Emit(bridle.TurnError{
			Err:   fmt.Errorf("claudesdk: retrying in %v (attempt %d/%d): %w", delay, attempt+1, maxRetries, err),
			Stage: bridle.TurnErrorStageRetry,
		})
		select {
		case <-ctx.Done():
			return bridle.ProviderResult{}, ctx.Err()
		case <-time.After(delay):
		}
	}
}

// turnState accumulates what the sidecar's event stream reports across
// one runTurnOnce call.
type turnState struct {
	finalText      string
	toolCalls      []bridle.ToolInvocation // custom calls, for the SESSION LOG only (spec: observability); ProviderResult.ToolCalls stays empty, see package doc
	thinkingBlocks []bridle.ThinkingBlock
	sessionDelta   []bridle.SessionEvent
	usage          bridle.Usage
	stopReason     bridle.StopReason
	resolvedModel  string
	sessionID      string // echoed by the sidecar's "done" event (spec §7 invariant 3)
	stepCount      int
	gotDone        bool
	refusal        bool
	providerErr    *bridle.ProviderError // set by an "error" event; terminal
}

func (p *Provider) runTurnOnce(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	sidecarPath := p.SidecarPath
	if sidecarPath == "" {
		sidecarPath = "bridle-claude-sidecar"
	}

	init := p.buildInit(req)
	// Marshal before spawning: init is built entirely from strings/JSON-
	// safe fields lowered from ProviderRequest, so this can't realistically
	// fail, but failing fast here (before there's a process to clean up)
	// is simpler than unwinding a live subprocess on a marshal error.
	initLine, err := json.Marshal(init)
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("claudesdk: marshal init: %w", err)
	}

	cmd := exec.Command(sidecarPath) //nolint:gosec // path is operator-configured, same trust model as claudecode.ClaudePath
	if req.Cwd != "" {
		cmd.Dir = req.Cwd
	}
	// Scrub ambient auth-erosion vars UNCONDITIONALLY (NEX-745 review
	// gate, MED) — regardless of whether ProviderEnv is set — before
	// overlaying ProviderEnv (which carries the authoritative
	// CLAUDE_CODE_OAUTH_TOKEN). See scrubAuthEnv's doc for why.
	cmd.Env = subprocess.MergeEnv(scrubAuthEnv(os.Environ()), req.ProviderEnv)
	// Process-group spawn (NEX-745 review gate, HIGH): the sidecar spawns
	// the real `claude` CLI as its OWN child (a grandchild of bridle), and
	// index.ts registers no SIGTERM handler, so a single-PID signal (the
	// old behavior — see claudecode/codexcli/geminicli, which still use
	// it unchanged) would leave that grandchild orphaned on cancellation.
	// SetPgid + WatchCancelGroup below signal the whole process group
	// instead. Opt-in per subprocess.SetPgid's doc: no other provider is
	// affected.
	subprocess.SetPgid(cmd)

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("claudesdk: stdin pipe: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("claudesdk: stdout pipe: %w", err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return bridle.ProviderResult{}, classifyStartError(sidecarPath, err)
	}

	procExited := make(chan struct{})
	go subprocess.WatchCancelGroup(ctx, cmd, procExited, subprocess.TermSignal())

	if _, err := stdinPipe.Write(append(initLine, '\n')); err != nil {
		// Write failed (sidecar died before reading its init line, or
		// never started reading). Kill defensively; the normal
		// cmd.Wait()/stderr-classification path below still runs and
		// surfaces the real cause.
		_ = cmd.Process.Kill()
	}

	state := &turnState{}
	streamDone := make(chan struct{})
	var streamErr error
	go func() {
		defer close(streamDone)
		streamErr = p.pumpEvents(ctx, req, stdoutPipe, stdinPipe, sink, state)
	}()

	startTime := time.Now()
	<-streamDone
	_ = stdinPipe.Close()
	waitErr := cmd.Wait()
	runtime := time.Since(startTime)
	close(procExited)

	stderrStr := stderr.String()

	if ctx.Err() != nil {
		// Clean interrupt: partial ProviderResult, matches the existing
		// harness contract (spec §7 — "ctx cancel → clean interrupt,
		// partial ProviderResult returned").
		return partialResult(state, bridle.StopReasonAborted), nil
	}

	if state.providerErr != nil {
		return partialResult(state, bridle.StopReasonError), state.providerErr
	}

	if streamErr != nil {
		return partialResult(state, bridle.StopReasonError), fmt.Errorf("claudesdk: %w [runtime=%s stderr=%q]", streamErr, runtime.Round(time.Millisecond), strings.TrimSpace(stderrStr))
	}

	if waitErr != nil {
		pe := classifyProviderError(stderrStr, waitErr)
		pe.Err = fmt.Errorf("%w [runtime=%s]", waitErr, runtime.Round(time.Millisecond))
		return partialResult(state, bridle.StopReasonError), pe
	}

	if !state.gotDone {
		return partialResult(state, bridle.StopReasonError), fmt.Errorf("claudesdk: sidecar exited without a done event [runtime=%s stderr=%q]", runtime.Round(time.Millisecond), strings.TrimSpace(stderrStr))
	}

	if state.refusal {
		return partialResult(state, bridle.StopReasonError), &bridle.ProviderError{
			Kind:    bridle.ProviderErrorRefusal,
			Message: "claudesdk: model refused the turn (stop_reason=refusal)",
		}
	}

	// Session resume-id invariant (spec §7 invariant 3): the SDK's
	// echoed session id should match what we asked it to resume. A
	// mismatch doesn't fail the turn (the content is still real) but is
	// surfaced as a warning — the same posture as claudecode's
	// stderr-output non-fatal surfacing.
	if req.Session.ID != "" && !req.Session.New && state.sessionID != "" && state.sessionID != req.Session.ID {
		sink.Emit(bridle.TurnError{
			Err:   fmt.Errorf("claudesdk: sidecar echoed session id %q, requested resume of %q", state.sessionID, req.Session.ID),
			Stage: bridle.TurnErrorStageStderrOutput,
		})
	}

	if stderrStr != "" {
		sink.Emit(bridle.TurnError{
			Err:   fmt.Errorf("claudesdk: stderr output: %s", strings.TrimSpace(stderrStr)),
			Stage: bridle.TurnErrorStageStderrOutput,
		})
	}

	return providerResult(state), nil
}

// scrubbedEnvKeys lists ambient env vars that must never reach the
// sidecar unfiltered (NEX-745 review gate, MED — auth-policy erosion).
// The vendored @anthropic-ai/claude-agent-sdk's auth precedence ranks
// ANTHROPIC_API_KEY (and ANTHROPIC_AUTH_TOKEN) ABOVE
// CLAUDE_CODE_OAUTH_TOKEN. Since a dual-lane bridle deployment (the
// direct claude-api lane reads ANTHROPIC_API_KEY from its own env in the
// SAME process) can easily have ANTHROPIC_API_KEY set in os.Environ(),
// leaving it in the sidecar's inherited env would make a claudesdk turn
// silently fall back to METERED api-key billing — defeating the entire
// premise of this subscription-token lane and failing spec §10.2's
// "no ANTHROPIC_API_KEY in env" acceptance criterion. Scrubbed
// unconditionally: the default funnel/subscription posture always wins;
// there is no v1 opt-in to api-key mode for this provider (an operator
// who wants that should use provider/claude instead, which already
// reads ANTHROPIC_API_KEY as its documented auth path).
var scrubbedEnvKeys = []string{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}

// scrubAuthEnv returns a copy of env with every KEY=VALUE entry whose
// KEY is in scrubbedEnvKeys removed. env is left unmodified.
func scrubAuthEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, kv := range env {
		skip := false
		for _, k := range scrubbedEnvKeys {
			if strings.HasPrefix(kv, k+"=") {
				skip = true
				break
			}
		}
		if !skip {
			out = append(out, kv)
		}
	}
	return out
}

// buildInit lowers a bridle.ProviderRequest into the sidecar's init line.
func (p *Provider) buildInit(req bridle.ProviderRequest) sidecarInit {
	init := sidecarInit{
		Type:               "init",
		Version:            protocolVersion,
		Prompt:             subprocess.LastUserPrompt(req.Messages),
		Model:              req.Model,
		MaxTurns:           req.MaxSteps,
		Cwd:                req.Cwd,
		Mode:               string(p.Mode),
		BeforeToolCallGate: true,
	}
	// Only the append system-prompt lane is offered in v1 (spec §5): the
	// claude-code preset is always the base; SystemPromptReplace isn't
	// supported yet and is silently ignored (same posture other
	// providers take toward fields their wire doesn't support — see
	// ProviderRequest field docs).
	if req.AppendSystemPrompt != "" {
		init.SystemPromptAppend = req.AppendSystemPrompt
	}

	if req.Session.ID != "" {
		if req.Session.New {
			init.SessionID = req.Session.ID
		} else {
			init.Resume = req.Session.ID
		}
	}

	if p.Mode == ModeAgent {
		init.AllowedTools = p.AllowedTools
		init.DisallowedTools = p.DisallowedTools
	}

	if len(req.Tools) > 0 {
		init.Tools = make([]sidecarToolDef, len(req.Tools))
		for i, t := range req.Tools {
			init.Tools[i] = sidecarToolDef{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema}
		}
	}

	if len(p.ExtraOpts) > 0 {
		init.ExtraOpts = p.ExtraOpts
	}

	return init
}

// pumpEvents reads the sidecar's JSON-lines event stream, dispatches
// each event to sink / state, and services custom tool calls via
// req.ToolExecutor — writing the tool_result back to the sidecar's
// stdin before continuing to read. Returns when the stream reaches EOF
// (the sidecar closed stdout, normally right after "done"/"error") or a
// read/dispatch error occurs.
func (p *Provider) pumpEvents(ctx context.Context, req bridle.ProviderRequest, stdout io.Reader, stdin io.Writer, sink bridle.EventSink, state *turnState) error {
	return subprocess.ScanJSONLines(stdout, func(line []byte) {
		var ev sidecarEvent
		if jsonErr := json.Unmarshal(line, &ev); jsonErr != nil {
			return // malformed line — skip, don't fail the turn (claudecode precedent)
		}

		switch ev.Type {
		case "text_delta":
			if ev.Text != "" {
				bridle.EmitAssistantText(sink, &state.finalText, &state.sessionDelta, providerID, ev.Text)
			}

		case "thinking_delta":
			// NEX-320: carried as a completed ThinkingBlock, not folded
			// into FinalText — the harness threads ThinkingBlocks into
			// the next turn's ProviderMessage so the Anthropic API sees
			// them round-tripped.
			if ev.Text != "" {
				state.thinkingBlocks = append(state.thinkingBlocks, bridle.ThinkingBlock{
					Type:      "thinking",
					Thinking:  ev.Text,
					Signature: ev.Signature,
				})
			}

		case "tool_call":
			// Custom tool call: bridle must execute it via the
			// harness-supplied ToolExecutor (NEX-745 §4) before the
			// sidecar's query() is allowed to continue.
			p.serviceToolCall(ctx, ev, req, sink, stdin, state)

		case "native_tool":
			// Observe-only (spec §5): the CLI's own native tool ran
			// inside the sidecar; bridle only watches. Sink-only, same
			// posture as claudecode's tool observation (no hook access
			// from inside a Provider — hooks are Harness-internal).
			sink.Emit(bridle.ToolCallStart{ID: ev.ID, Name: ev.Name, Args: ev.Args})
			tcr := bridle.ToolCallResult{ID: ev.ID, Result: ev.Result}
			if ev.IsError {
				tcr.Err = string(ev.Result)
			}
			sink.Emit(tcr)
			state.stepCount++
			sink.Emit(bridle.StepBoundary{Step: state.stepCount})
			state.sessionDelta = append(state.sessionDelta, bridle.SessionEvent{
				Provider:   providerID,
				Role:       bridle.RoleTool,
				Content:    string(ev.Result),
				ToolCallID: ev.ID,
			})

		case "usage":
			state.usage.InputTokens = ev.InputTokens
			state.usage.OutputTokens = ev.OutputTokens
			state.usage.CacheReadInputTokens = ev.CacheReadInputTokens
			state.usage.CacheCreationInputTokens = ev.CacheCreationInputTokens

		case "done":
			state.gotDone = true
			state.sessionID = ev.SessionID
			if ev.Model != "" {
				state.resolvedModel = ev.Model
			}
			if ev.StopReason == "refusal" {
				state.refusal = true
			}
			state.stopReason = normalizeStopReason(ev.StopReason)

		case "error":
			state.providerErr = &bridle.ProviderError{
				Kind:    classifyWireErrorClass(ev.Class),
				Message: "claudesdk: " + ev.Message,
			}
		}
	})
}

// serviceToolCall executes one custom tool_call event via
// req.ToolExecutor and writes the tool_result back to the sidecar's
// stdin. A nil ToolExecutor (misconfigured caller — the harness always
// wires one for a subprocess-stream+SupportsCustomTools provider, see
// run.go) or an executor error is surfaced as a provider-class error and
// stops the exchange; the sidecar process is left to the caller's
// context-cancellation / process-exit handling.
func (p *Provider) serviceToolCall(ctx context.Context, ev sidecarEvent, req bridle.ProviderRequest, sink bridle.EventSink, stdin io.Writer, state *turnState) {
	if req.ToolExecutor == nil {
		state.providerErr = &bridle.ProviderError{
			Kind:    bridle.ProviderErrorConfig,
			Message: "claudesdk: sidecar requested a custom tool call but no ToolExecutor is configured",
		}
		return
	}

	call := bridle.ToolCall{ID: ev.ID, Name: ev.Name, Args: ev.Args}
	result, err := req.ToolExecutor.Execute(ctx, call)

	reply := sidecarToolResult{Type: "tool_result", ID: ev.ID}
	if err != nil {
		reply.IsError = true
		reply.Content = json.RawMessage(`"` + jsonEscape(err.Error()) + `"`)
	} else if result.Err != "" {
		reply.IsError = true
		reply.Content = json.RawMessage(`"` + jsonEscape(result.Err) + `"`)
	} else {
		reply.Content = result.Result
		if len(reply.Content) == 0 {
			reply.Content = json.RawMessage(`null`)
		}
	}

	state.stepCount++
	state.toolCalls = append(state.toolCalls, bridle.ToolInvocation{
		ID: ev.ID, Name: ev.Name, Args: ev.Args, Result: reply.Content, Err: result.Err,
	})
	state.sessionDelta = append(state.sessionDelta, bridle.SessionEvent{
		Provider:   providerID,
		Role:       bridle.RoleTool,
		Content:    string(reply.Content),
		ToolCallID: ev.ID,
	})
	sink.Emit(bridle.StepBoundary{Step: state.stepCount})

	line, marshalErr := json.Marshal(reply)
	if marshalErr != nil {
		state.providerErr = &bridle.ProviderError{Kind: bridle.ProviderErrorSubprocessExit, Message: "claudesdk: marshal tool_result: " + marshalErr.Error()}
		return
	}
	if _, writeErr := stdin.Write(append(line, '\n')); writeErr != nil {
		state.providerErr = &bridle.ProviderError{Kind: bridle.ProviderErrorSubprocessExit, Message: "claudesdk: write tool_result: " + writeErr.Error()}
	}
}

// jsonEscape is a minimal string escaper for folding a Go error string
// into a JSON string literal without pulling in a full marshal round
// trip. Only used for the synthetic error-content payload above.
func jsonEscape(s string) string {
	b, _ := json.Marshal(s)
	if len(b) < 2 {
		return ""
	}
	return string(b[1 : len(b)-1])
}

// normalizeStopReason maps the sidecar's stop_reason string to bridle's
// canonical StopReason. "refusal" still normalizes to StopReasonError
// here for completeness, but runTurnOnce intercepts state.refusal before
// this value is ever surfaced in a successful ProviderResult — a
// refusal turn always returns via the ProviderErrorRefusal path instead.
func normalizeStopReason(raw string) bridle.StopReason {
	switch raw {
	case "max_turns":
		return bridle.StopReasonMaxSteps
	case "refusal", "error":
		return bridle.StopReasonError
	case "":
		return bridle.StopReasonModelDone
	default:
		return bridle.StopReasonModelDone
	}
}

// classifyWireErrorClass maps the sidecar's "error" event class field
// (spec §7: auth | rate_limit | provider | refusal) to a bridle
// ProviderErrorKind.
func classifyWireErrorClass(class string) bridle.ProviderErrorKind {
	switch class {
	case "auth":
		return bridle.ProviderErrorAuthFailed
	case "rate_limit":
		return bridle.ProviderErrorRateLimit
	case "refusal":
		return bridle.ProviderErrorRefusal
	default:
		return bridle.ProviderErrorSubprocessExit
	}
}

func partialResult(state *turnState, stopReason bridle.StopReason) bridle.ProviderResult {
	return bridle.ProviderResult{
		FinalText: state.finalText,
		// ToolCalls stays empty — see package doc: custom calls already
		// executed via ToolExecutor, so returning them here would make
		// run.go's per-round loop re-run them.
		StepCount:      state.stepCount,
		Usage:          state.usage,
		StopReason:     stopReason,
		SessionDelta:   state.sessionDelta,
		ResolvedModel:  state.resolvedModel,
		ThinkingBlocks: state.thinkingBlocks,
	}
}

func providerResult(state *turnState) bridle.ProviderResult {
	return partialResult(state, state.stopReason)
}

// isRetryable reports whether a provider error is transient and should
// be retried, mirroring claudecode's isRetryable. Refusal and auth
// failures are explicitly NOT retryable — retrying with the same
// credentials/turn produces the same result.
func isRetryable(err error) bool {
	pe := &bridle.ProviderError{}
	if !errors.As(err, &pe) {
		return false
	}
	switch pe.Kind {
	case bridle.ProviderErrorRateLimit,
		bridle.ProviderErrorServerError,
		bridle.ProviderErrorNetworkError,
		bridle.ProviderErrorTimeout:
		return true
	}
	return false
}

// errorPatterns is claudesdk's ordered stderr classification table,
// mirroring claudecode's — the sidecar's own stderr (Node/SDK crash
// output, not the JSON-lines wire) is the only place these patterns are
// matched; the normal error path is the wire's "error" event
// (classifyWireErrorClass above).
var errorPatterns = []subprocess.Pattern{
	{
		Kind:     bridle.ProviderErrorAuthFailed,
		Patterns: []string{"not logged in", "authentication_failed", "run /login", "invalid_api_key", "oauth"},
		Message:  "claudesdk: authentication failed — check CLAUDE_CODE_OAUTH_TOKEN / claude login",
	},
	{
		Kind:     bridle.ProviderErrorConfig,
		Patterns: []string{"cannot find module", "enoent", "command not found"},
		Message:  "claudesdk: sidecar failed to start — check SidecarPath / node install",
	},
}

// classifyProviderError inspects the sidecar's stderr and classifies the
// subprocess error, mirroring claudecode.classifyProviderError.
func classifyProviderError(stderr string, waitErr error) *bridle.ProviderError {
	if kind, msg, ok := subprocess.ClassifyWithFallback(stderr, "claudesdk", errorPatterns); ok {
		return &bridle.ProviderError{Kind: kind, Message: msg, Err: waitErr}
	}
	msg := "claudesdk: sidecar exited with error"
	if s := strings.TrimSpace(stderr); s != "" {
		msg += " (stderr: " + s + ")"
	}
	return &bridle.ProviderError{Kind: bridle.ProviderErrorSubprocessExit, Message: msg, Err: waitErr}
}

// classifyStartError classifies a failure to even start the sidecar
// process (binary missing, not executable, etc.) — always a config
// error, never retryable.
func classifyStartError(sidecarPath string, err error) *bridle.ProviderError {
	return &bridle.ProviderError{
		Kind:    bridle.ProviderErrorConfig,
		Message: fmt.Sprintf("claudesdk: failed to start sidecar %q: %v", sidecarPath, err),
		Err:     err,
	}
}
