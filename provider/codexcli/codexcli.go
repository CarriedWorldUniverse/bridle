// Package codexcli implements the bridle Provider interface using the
// Codex CLI in non-interactive mode (`codex exec --json`).
//
// Category: subprocess-stream. The CLI manages its own agentic loop;
// bridle parses the stdout JSONL event stream. BeforeToolCall does not
// fire; AfterToolCall fires after each parsed command_execution result
// (observe-only).
//
// Session continuity: a fresh turn uses `codex exec <prompt>` and records
// the emitted thread_id as a system SessionEvent. A continuing turn with a
// non-empty Session.ID uses `codex exec resume <id> <prompt>`.
package codexcli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/internal/subprocess"
)

const providerID = bridle.ProviderCodexCLI

// Provider implements bridle.Provider by shelling out to the Codex CLI.
type Provider struct {
	// CodexPath is the path to the codex binary. Defaults to "codex" (PATH lookup).
	CodexPath string
	// Sandbox maps to global `--sandbox` (read-only, workspace-write,
	// danger-full-access). Empty lets the CLI/config choose.
	Sandbox string
	// ApprovalPolicy maps to global `--ask-for-approval`. Default from
	// New() is "never" so headless turns do not block on prompts.
	ApprovalPolicy string
	// Profile maps to global `--profile`.
	Profile string
	// ExtraConfig appends repeated global `--config key=value` overrides.
	ExtraConfig []string
	// ExtraArgs are appended after the exec/resume options and before the prompt.
	ExtraArgs []string
	// Ephemeral passes `--ephemeral`, disabling Codex session persistence.
	Ephemeral bool
	// SkipGitRepoCheck passes `--skip-git-repo-check`.
	SkipGitRepoCheck bool
	// IgnoreUserConfig passes `--ignore-user-config`.
	IgnoreUserConfig bool
	// IgnoreRules passes `--ignore-rules`.
	IgnoreRules bool
	// BypassApprovalsAndSandbox passes
	// `--dangerously-bypass-approvals-and-sandbox`. Intended only for
	// externally sandboxed automation.
	BypassApprovalsAndSandbox bool
}

// New returns a codexcli Provider with headless-safe defaults.
func New() *Provider {
	return &Provider{
		CodexPath:      "codex",
		ApprovalPolicy: "never",
		// A codex aspect runs inside a trust boundary the funnel controls
		// (a trusted host or a k8s pod), so codex's own internal sandbox is
		// redundant — and its default (workspace-write) blocks network,
		// which broke aspect git push (NEX-433 follow-up). The surrounding
		// environment is the real sandbox; disable codex's. Override the
		// field for stricter environments.
		Sandbox:          "danger-full-access",
		SkipGitRepoCheck: true,
	}
}

func (p *Provider) Name() bridle.ProviderID { return providerID }

func (p *Provider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{
		Category:               bridle.CategorySubprocessStream,
		SupportsCustomTools:    false,
		SupportsBeforeToolCall: false,
		SupportsAfterToolCall:  true,
		SupportsMCP:            true,
	}
}

// RunTurn invokes the Codex CLI and streams its output as bridle events.
// Tool calls are executed by Codex; bridle's ToolRunner is not called.
// Cancellation via ctx sends SIGTERM then SIGKILL after a 5s grace period.
//
// Session-resume robustness (NEX-588): when a continuing turn resumes a
// session (`codex exec resume <id>`) and codex reports the thread is
// missing/corrupt, the turn degrades to a FRESH session (drop the
// resume, re-run) with a logged TurnErrorStageResumeFallback warning
// rather than failing the turn. The fresh attempt's content is what the
// caller gets — the lost prior context is unrecoverable anyway. A
// transient/auth resume error is NOT treated this way: it propagates so
// it can be retried against the same session (a builder mid-task must
// not silently lose context).
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	resuming := req.Session.ID != "" && !req.Session.New
	result, err := p.runTurnOnce(ctx, req, sink)
	if err == nil || !resuming || ctx.Err() != nil {
		return result, err
	}
	// Only degrade to fresh when the resume genuinely failed because the
	// session is missing/corrupt — never on auth/rate/network/crash,
	// which must surface as-is so they can be retried/diagnosed.
	var pe *bridle.ProviderError
	if !errors.As(err, &pe) || !subprocess.IsResumeNotFound(pe.Error()) {
		return result, err
	}
	sink.Emit(bridle.TurnError{
		Err: fmt.Errorf("codexcli: resume of session %q failed (missing/corrupt) — falling back to a fresh session; prior context is lost: %w",
			req.Session.ID, err),
		Stage: bridle.TurnErrorStageResumeFallback,
	})
	fresh := req
	fresh.Session.New = true
	return p.runTurnOnce(ctx, fresh, sink)
}

func (p *Provider) runTurnOnce(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	codexPath := p.CodexPath
	if codexPath == "" {
		codexPath = "codex"
	}

	args := p.buildCLIArgs(req)
	cmd := exec.Command(codexPath, args...)
	if req.Cwd != "" {
		cmd.Dir = req.Cwd
	}
	cmd.Stdin = strings.NewReader("")
	if len(req.ProviderEnv) > 0 {
		cmd.Env = subprocess.MergeEnv(os.Environ(), req.ProviderEnv)
	}

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("codexcli: pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("codexcli: start: %w", err)
	}

	// Cancel watcher: graceful signal + grace period + Kill. procExited
	// is closed AFTER cmd.Wait() returns; see subprocess.WatchCancel for
	// the full contract. codexcli keeps its own sigterm() (os.Kill on
	// Windows, vs os.Interrupt for the other CLI providers).
	procExited := make(chan struct{})
	go subprocess.WatchCancel(ctx, cmd, procExited, sigterm())

	streamDone := make(chan struct{})
	var result bridle.ProviderResult
	var parseErr error
	go func() {
		defer close(streamDone)
		result, parseErr = parseStream(stdoutPipe, sink)
	}()

	// Drain the stdout pipe to EOF BEFORE cmd.Wait(). StdoutPipe's
	// contract is that Wait closes the pipe FD once the process exits;
	// calling Wait while the scan goroutine is still reading races the
	// FD close and can surface a spurious "file already closed" (and, on
	// a fast exit, drop unread lines — including the terminal result
	// event). Waiting for streamDone first lets the goroutine read to a
	// clean EOF (the process closing stdout on exit) before we reap.
	<-streamDone
	waitErr := cmd.Wait()
	close(procExited)

	stderrStr := stderr.String()
	if waitErr != nil && parseErr == nil {
		if ctx.Err() != nil {
			result.StopReason = bridle.StopReasonAborted
		} else if result.FinalText != "" || len(result.ToolCalls) > 0 {
			result.StopReason = bridle.StopReasonProcessExit
			sink.Emit(bridle.TurnError{
				Err:   fmt.Errorf("codexcli: subprocess exited non-zero with partial content: %w (stderr: %s)", waitErr, strings.TrimSpace(stderrStr)),
				Stage: bridle.TurnErrorStageSubprocessExitPartial,
			})
		} else {
			pe := classifyProviderError(stderrStr, waitErr)
			sink.Emit(bridle.TurnError{Err: pe, Stage: bridle.TurnErrorStage(pe.Kind)})
			return bridle.ProviderResult{}, pe
		}
	}

	if stderrStr != "" && waitErr == nil && parseErr == nil && ctx.Err() == nil {
		sink.Emit(bridle.TurnError{
			Err:   fmt.Errorf("codexcli: stderr output: %s", strings.TrimSpace(stderrStr)),
			Stage: bridle.TurnErrorStageStderrOutput,
		})
	}

	if parseErr == nil && result.StopReason == "" && ctx.Err() == nil {
		msg := "codexcli: stream ended without turn.completed event"
		if stderrStr != "" {
			msg += " (stderr: " + strings.TrimSpace(stderrStr) + ")"
		}
		parseErr = fmt.Errorf("%s", msg)
		sink.Emit(bridle.TurnError{Err: parseErr, Stage: bridle.TurnErrorStageStreamTruncated})
	}

	if parseErr != nil && stderrStr != "" {
		parseErr = fmt.Errorf("codexcli: %w (stderr: %s)", parseErr, strings.TrimSpace(stderrStr))
	}
	return result, parseErr
}

func (p *Provider) buildCLIArgs(req bridle.ProviderRequest) []string {
	args := []string{}
	if p.Profile != "" {
		args = append(args, "--profile", p.Profile)
	}
	if p.Sandbox != "" {
		args = append(args, "--sandbox", p.Sandbox)
	}
	if p.ApprovalPolicy != "" {
		args = append(args, "--ask-for-approval", p.ApprovalPolicy)
	}
	if p.BypassApprovalsAndSandbox {
		args = append(args, "--dangerously-bypass-approvals-and-sandbox")
	}
	for _, cfg := range p.ExtraConfig {
		args = append(args, "--config", cfg)
	}
	// Wire bridle's MCP servers into codex as per-invocation --config
	// overrides (mcp_servers.<name>.*). codex exec connects to these and
	// exposes their tools to the model as mcp__<name>.<tool> — the path by
	// which a codex aspect gets the funnel-supplied nexus tools (comms,
	// jira, ledger, auth, …), even though those execute over the broker WS.
	if req.MCP != nil {
		args = append(args, mcpConfigArgs(req.MCP.Servers)...)
	}
	if req.Cwd != "" {
		args = append(args, "--cd", req.Cwd)
	}

	args = append(args, "exec")
	if req.Session.ID != "" && !req.Session.New {
		args = append(args, "resume", req.Session.ID)
	}

	args = append(args, "--json")
	if p.Ephemeral {
		args = append(args, "--ephemeral")
	}
	if p.SkipGitRepoCheck {
		args = append(args, "--skip-git-repo-check")
	}
	if p.IgnoreUserConfig {
		args = append(args, "--ignore-user-config")
	}
	if p.IgnoreRules {
		args = append(args, "--ignore-rules")
	}
	if req.Model != "" && req.Model != "default" {
		args = append(args, "--model", req.Model)
	}
	args = append(args, p.ExtraArgs...)

	prompt := subprocess.LastUserPrompt(req.Messages)
	if prompt != "" {
		args = append(args, prompt)
	}
	return args
}

// mcpConfigArgs translates bridle MCP server specs into codex
// `--config mcp_servers.<name>.*` arguments. Each value is TOML-encoded
// (codex parses the value portion of --config as TOML). stdio servers map
// to command/args/env; http_sse servers map to url.
func mcpConfigArgs(servers []bridle.MCPServerSpec) []string {
	var out []string
	add := func(key, tomlVal string) {
		out = append(out, "--config", key+"="+tomlVal)
	}
	for _, s := range servers {
		if s.Name == "" {
			continue
		}
		base := "mcp_servers." + tomlBareKey(s.Name)
		if s.Transport == bridle.MCPTransportHTTPSSE {
			if s.URL == "" {
				continue
			}
			add(base+".url", tomlString(s.URL))
			continue
		}
		// stdio (default)
		if len(s.Command) == 0 {
			continue
		}
		add(base+".command", tomlString(s.Command[0]))
		if len(s.Command) > 1 {
			add(base+".args", tomlStringArray(s.Command[1:]))
		}
		if len(s.Env) > 0 {
			add(base+".env", tomlStringTable(s.Env))
		}
	}
	return out
}

// tomlString returns a TOML basic-string literal for s.
func tomlString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

func tomlStringArray(ss []string) string {
	parts := make([]string, len(ss))
	for i, s := range ss {
		parts[i] = tomlString(s)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func tomlStringTable(m map[string]string) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys) // deterministic args (tests + prefix-cache stability)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, tomlBareKey(k)+"="+tomlString(m[k]))
	}
	return "{" + strings.Join(parts, ",") + "}"
}

// tomlBareKey returns k as a TOML key — bare when it matches the bare-key
// grammar (A-Za-z0-9_-), otherwise a quoted string. Server + env-var names
// are normally bare-safe.
func tomlBareKey(k string) string {
	if k == "" {
		return `""`
	}
	for _, r := range k {
		if !(r == '-' || r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')) {
			return tomlString(k)
		}
	}
	return k
}

func parseStream(r io.Reader, sink bridle.EventSink) (bridle.ProviderResult, error) {
	var (
		finalText    string
		toolCalls    []bridle.ToolInvocation
		sessionDelta []bridle.SessionEvent
		usage        bridle.Usage
		stopReason   bridle.StopReason
		stepCount    int
		gotDone      bool
	)

	pendingCalls := map[string]bridle.ToolCallStart{}

	perLine := func(line []byte) {
		if line[0] != '{' {
			return
		}

		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(line, &head); err != nil {
			return
		}

		switch head.Type {
		case "thread.started":
			var ev struct {
				ThreadID string `json:"thread_id"`
			}
			if err := json.Unmarshal(line, &ev); err == nil {
				raw, _ := json.Marshal(map[string]any{
					"type":      head.Type,
					"thread_id": ev.ThreadID,
				})
				sessionDelta = append(sessionDelta, bridle.SessionEvent{
					Provider: providerID,
					Role:     bridle.RoleSystem,
					RawJSON:  raw,
				})
			}

		case "item.started":
			var ev struct {
				Item codexItem `json:"item"`
			}
			if err := json.Unmarshal(line, &ev); err == nil && ev.Item.Type == "command_execution" {
				args, _ := json.Marshal(map[string]string{"command": ev.Item.Command})
				tc := bridle.ToolCallStart{ID: ev.Item.ID, Name: "command_execution", Args: args}
				sink.Emit(tc)
				pendingCalls[ev.Item.ID] = tc
				raw, _ := json.Marshal(map[string]string{
					"type":    "command_execution",
					"command": ev.Item.Command,
				})
				sessionDelta = append(sessionDelta, bridle.SessionEvent{
					Provider: providerID,
					Role:     bridle.RoleAssistant,
					RawJSON:  raw,
				})
			}

		case "item.completed":
			var ev struct {
				Item codexItem `json:"item"`
			}
			if err := json.Unmarshal(line, &ev); err != nil {
				return
			}
			switch ev.Item.Type {
			case "agent_message":
				bridle.EmitAssistantText(sink, &finalText, &sessionDelta, providerID, ev.Item.Text)
			case "command_execution":
				resultJSON, _ := json.Marshal(map[string]any{
					"output":    ev.Item.AggregatedOutput,
					"exit_code": ev.Item.ExitCode,
					"status":    ev.Item.Status,
				})
				tcr := bridle.ToolCallResult{ID: ev.Item.ID, Result: resultJSON}
				if ev.Item.ExitCode != nil && *ev.Item.ExitCode != 0 {
					tcr.Err = fmt.Sprintf("exit code %d", *ev.Item.ExitCode)
				}
				sink.Emit(tcr)

				if start, ok := pendingCalls[ev.Item.ID]; ok {
					toolCalls = append(toolCalls, bridle.ToolInvocation{
						ID:     start.ID,
						Name:   start.Name,
						Args:   start.Args,
						Result: resultJSON,
						Err:    tcr.Err,
					})
					delete(pendingCalls, ev.Item.ID)
					stepCount++
					sink.Emit(bridle.StepBoundary{Step: stepCount})
				}
				sessionDelta = append(sessionDelta, bridle.SessionEvent{
					Provider: providerID,
					Role:     bridle.RoleTool,
					Content:  string(resultJSON),
				})
			}

		case "turn.completed":
			var ev struct {
				Usage struct {
					InputTokens           int `json:"input_tokens"`
					CachedInputTokens     int `json:"cached_input_tokens"`
					OutputTokens          int `json:"output_tokens"`
					ReasoningOutputTokens int `json:"reasoning_output_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(line, &ev); err == nil {
				usage.InputTokens = ev.Usage.InputTokens
				usage.CacheReadInputTokens = ev.Usage.CachedInputTokens
				usage.OutputTokens = ev.Usage.OutputTokens
				stopReason = bridle.StopReasonModelDone
				gotDone = true
			}

		case "turn.failed":
			stopReason = bridle.StopReasonError
			gotDone = true
		}
	}

	if err := subprocess.ScanJSONLines(r, perLine); err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("codexcli: stream read: %w", err)
	}

	if !gotDone {
		return bridle.ProviderResult{
			FinalText:    finalText,
			ToolCalls:    toolCalls,
			StepCount:    stepCount,
			Usage:        usage,
			SessionDelta: sessionDelta,
		}, nil
	}

	return bridle.ProviderResult{
		FinalText:    finalText,
		ToolCalls:    toolCalls,
		StepCount:    stepCount,
		Usage:        usage,
		StopReason:   stopReason,
		SessionDelta: sessionDelta,
	}, nil
}

type codexItem struct {
	ID               string `json:"id"`
	Type             string `json:"type"`
	Text             string `json:"text"`
	Command          string `json:"command"`
	AggregatedOutput string `json:"aggregated_output"`
	ExitCode         *int   `json:"exit_code"`
	Status           string `json:"status"`
}

// errorPatterns is codexcli's ordered stderr classification table —
// the same groups, order and substrings as the previous inline switch
// (note: timeout before network, unlike claudecode). See
// subprocess.Pattern for matching semantics.
var errorPatterns = []subprocess.Pattern{
	{
		Kind:     bridle.ProviderErrorAuthFailed,
		Patterns: []string{"not logged in", "authentication", "login"},
		Message:  "codexcli: authentication failed. Check Codex login or OPENAI_API_KEY",
	},
	{
		Kind:     bridle.ProviderErrorRateLimit,
		Patterns: []string{"rate limit", "rate_limit"},
		Message:  "codexcli: rate limited",
	},
	{
		Kind:     bridle.ProviderErrorTimeout,
		Patterns: []string{"timeout", "timed out"},
		Message:  "codexcli: request timed out",
	},
	{
		Kind:     bridle.ProviderErrorNetworkError,
		Patterns: []string{"connection refused", "no route to host", "connection reset"},
		Message:  "codexcli: network error connecting to provider",
	},
}

func classifyProviderError(stderr string, waitErr error) *bridle.ProviderError {
	// codexcli's own patterns first (CLI-accurate wording), then the
	// shared actionable table. The shared layer is why a codex 401
	// (expired auth.json / wrong endpoint) — which codex surfaces as a
	// bare "exit status 1" with only "401"/"unauthorized" in stderr —
	// now classifies as AUTH instead of the opaque subprocess-exit that
	// forced manual in-pod debugging (NEX-588).
	if kind, msg, ok := subprocess.ClassifyWithFallback(stderr, "codexcli", errorPatterns); ok {
		return &bridle.ProviderError{Kind: kind, Message: msg, Err: waitErr}
	}
	msg := "codexcli: subprocess exited with error"
	if s := strings.TrimSpace(stderr); s != "" {
		// Keep stderr in the message so downstream checks (resume-not-found
		// fallback) and operators see the real cause, not a bare exit code.
		msg += " (stderr: " + s + ")"
	}
	return &bridle.ProviderError{
		Kind:    bridle.ProviderErrorSubprocessExit,
		Message: msg,
		Err:     waitErr,
	}
}
