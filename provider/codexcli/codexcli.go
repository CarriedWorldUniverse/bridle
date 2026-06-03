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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
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
		CodexPath:        "codex",
		ApprovalPolicy:   "never",
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
		SupportsMCP:            false,
	}
}

// RunTurn invokes the Codex CLI and streams its output as bridle events.
// Tool calls are executed by Codex; bridle's ToolRunner is not called.
// Cancellation via ctx sends SIGTERM then SIGKILL after a 5s grace period.
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
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
		cmd.Env = mergeEnv(os.Environ(), req.ProviderEnv)
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

	procExited := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			_ = cmd.Process.Signal(sigterm())
			timer := time.NewTimer(5 * time.Second)
			defer timer.Stop()
			select {
			case <-timer.C:
				_ = cmd.Process.Kill()
			case <-procExited:
			}
		case <-procExited:
		}
	}()

	streamDone := make(chan struct{})
	var result bridle.ProviderResult
	var parseErr error
	go func() {
		defer close(streamDone)
		result, parseErr = parseStream(stdoutPipe, sink)
	}()

	waitErr := cmd.Wait()
	close(procExited)
	<-streamDone

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

	prompt := buildPrompt(req)
	if prompt != "" {
		args = append(args, prompt)
	}
	return args
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

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 || line[0] != '{' {
			continue
		}

		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(line, &head); err != nil {
			continue
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
				continue
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

	if err := scanner.Err(); err != nil && err != io.EOF {
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

func buildPrompt(req bridle.ProviderRequest) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			return req.Messages[i].Content
		}
	}
	return ""
}

func mergeEnv(base []string, overlay map[string]string) []string {
	if len(overlay) == 0 {
		return base
	}
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

func classifyProviderError(stderr string, waitErr error) *bridle.ProviderError {
	lower := strings.ToLower(stderr)
	switch {
	case strings.Contains(lower, "not logged in") || strings.Contains(lower, "authentication") || strings.Contains(lower, "login"):
		return &bridle.ProviderError{
			Kind:    bridle.ProviderErrorAuthFailed,
			Message: "codexcli: authentication failed. Check Codex login or OPENAI_API_KEY",
			Err:     waitErr,
		}
	case strings.Contains(lower, "rate limit") || strings.Contains(lower, "rate_limit"):
		return &bridle.ProviderError{
			Kind:    bridle.ProviderErrorRateLimit,
			Message: "codexcli: rate limited",
			Err:     waitErr,
		}
	case strings.Contains(lower, "timeout") || strings.Contains(lower, "timed out"):
		return &bridle.ProviderError{
			Kind:    bridle.ProviderErrorTimeout,
			Message: "codexcli: request timed out",
			Err:     waitErr,
		}
	case strings.Contains(lower, "connection refused") || strings.Contains(lower, "no route to host") || strings.Contains(lower, "connection reset"):
		return &bridle.ProviderError{
			Kind:    bridle.ProviderErrorNetworkError,
			Message: "codexcli: network error connecting to provider",
			Err:     waitErr,
		}
	}
	return &bridle.ProviderError{
		Kind:    bridle.ProviderErrorSubprocessExit,
		Message: "codexcli: subprocess exited with error",
		Err:     waitErr,
	}
}
