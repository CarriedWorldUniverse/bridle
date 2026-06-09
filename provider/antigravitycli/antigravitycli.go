// Package antigravitycli implements the bridle Provider interface using the
// antigravity CLI in headless mode (antigravity -p --output-format stream-json).
//
// Category: subprocess-stream. The CLI manages its own agentic loop;
// bridle parses the stdout event stream. BeforeToolCall does not fire;
// AfterToolCall fires after each parsed tool_use/tool_result pair (observe-only).
//
// Auth: the CLI itself decides — Google login/credentials. Bridle does not pass credentials.
//
// Session continuity: antigravity's --conversation flag resumes a previous conversation by ID.
// The Session.ID field is passed through to --conversation.
package antigravitycli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

const providerID = bridle.ProviderAntigravityCLI

// Provider implements bridle.Provider by shelling out to the antigravity CLI.
type Provider struct {
	// AntigravityPath is the path to the antigravity binary. Defaults to "antigravity" (PATH lookup).
	AntigravityPath string
	// Yolo passes --dangerously-skip-permissions so the CLI auto-approves tool calls (recommended for headless).
	Yolo bool
	// AllowedTools restricts which native tools the CLI may use.
	// Maps to --allowed-tools (if supported by the CLI).
	AllowedTools []string
	// ExtraArgs are appended verbatim to the antigravity invocation.
	ExtraArgs []string
}

// New returns an antigravitycli Provider with default settings (yolo on).
// Headless agent use almost always wants yolo on so that tool calls do not block on prompts.
func New() *Provider {
	return &Provider{AntigravityPath: "antigravity", Yolo: true}
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

// RunTurn invokes the antigravity CLI and streams its output as bridle events.
// Tool calls are executed by the CLI; bridle's ToolRunner is not called.
// Cancellation via ctx sends SIGTERM then SIGKILL after a 5s grace period.
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	antigravityPath := p.AntigravityPath
	if antigravityPath == "" {
		antigravityPath = "antigravity"
	}

	args := p.buildCLIArgs(req)

	cmd := exec.Command(antigravityPath, args...)
	if req.Cwd != "" {
		cmd.Dir = req.Cwd
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("antigravitycli: pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("antigravitycli: start: %w", err)
	}

	// Cancel watcher: SIGTERM + grace period + SIGKILL.
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
				// Process exited cleanly during grace period — no SIGKILL needed.
			}
		case <-procExited:
			// Natural exit — nothing to do.
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
	close(procExited) // signal the cancel watcher that the process is gone
	<-streamDone

	if waitErr != nil && parseErr == nil {
		if ctx.Err() != nil {
			result.StopReason = bridle.StopReasonAborted
		} else {
			sink.Emit(bridle.TurnError{Err: fmt.Errorf("antigravitycli: %w", waitErr), Stage: bridle.TurnErrorStageSubprocessExit})
			return bridle.ProviderResult{}, fmt.Errorf("antigravitycli: CLI error: %w (stderr: %s)", waitErr, stderr.String())
		}
	}

	if parseErr == nil && result.StopReason == "" && ctx.Err() == nil {
		parseErr = fmt.Errorf("antigravitycli: stream ended without result event")
		sink.Emit(bridle.TurnError{Err: parseErr, Stage: bridle.TurnErrorStageStreamTruncated})
	}

	return result, parseErr
}

// parseStream reads stream-json lines and maps them to bridle events + result.
func parseStream(r io.Reader, sink bridle.EventSink) (bridle.ProviderResult, error) {
	var (
		finalText    string
		toolCalls    []bridle.ToolInvocation
		sessionDelta []bridle.SessionEvent
		usage        bridle.Usage
		stopReason   bridle.StopReason
		stepCount    int
		gotResult    bool
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
		case "init":
			var ev struct {
				SessionID string `json:"session_id"`
				Model     string `json:"model"`
			}
			if err := json.Unmarshal(line, &ev); err == nil {
				raw, _ := json.Marshal(map[string]any{
					"session_id": ev.SessionID,
					"model":      ev.Model,
				})
				sessionDelta = append(sessionDelta, bridle.SessionEvent{
					Provider: providerID,
					Role:     bridle.RoleSystem,
					RawJSON:  raw,
				})
			}

		case "message":
			var ev struct {
				Role    string `json:"role"`
				Content string `json:"content"`
				Delta   bool   `json:"delta"`
			}
			if err := json.Unmarshal(line, &ev); err == nil {
				if ev.Role == "assistant" {
					bridle.EmitAssistantText(sink, &finalText, &sessionDelta, providerID, ev.Content)
				}
			}

		case "tool_use":
			var ev struct {
				ToolName   string          `json:"tool_name"`
				ToolID     string          `json:"tool_id"`
				Parameters json.RawMessage `json:"parameters"`
			}
			if err := json.Unmarshal(line, &ev); err == nil {
				tc := bridle.ToolCallStart{ID: ev.ToolID, Name: ev.ToolName, Args: ev.Parameters}
				sink.Emit(tc)
				pendingCalls[ev.ToolID] = tc
				raw, _ := json.Marshal(ev)
				sessionDelta = append(sessionDelta, bridle.SessionEvent{
					Provider: providerID,
					Role:     bridle.RoleAssistant,
					RawJSON:  raw,
				})
			}

		case "tool_result":
			var ev struct {
				ToolID string          `json:"tool_id"`
				Status string          `json:"status"`
				Result json.RawMessage `json:"result"`
			}
			if err := json.Unmarshal(line, &ev); err == nil {
				resultJSON := ev.Result
				if len(resultJSON) == 0 {
					resultJSON, _ = json.Marshal(map[string]any{"status": ev.Status})
				}
				sink.Emit(bridle.ToolCallResult{ID: ev.ToolID, Result: resultJSON})

				if start, ok := pendingCalls[ev.ToolID]; ok {
					toolCalls = append(toolCalls, bridle.ToolInvocation{
						ID:     start.ID,
						Name:   start.Name,
						Args:   start.Args,
						Result: resultJSON,
					})
					delete(pendingCalls, ev.ToolID)
					stepCount++
					sink.Emit(bridle.StepBoundary{Step: stepCount})
				}
				sessionDelta = append(sessionDelta, bridle.SessionEvent{
					Provider: providerID,
					Role:     bridle.RoleTool,
					Content:  string(resultJSON),
				})
			}

		case "result":
			var ev struct {
				Status string `json:"status"`
				Stats  struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
				} `json:"stats"`
			}
			if err := json.Unmarshal(line, &ev); err == nil {
				usage.InputTokens = ev.Stats.InputTokens
				usage.OutputTokens = ev.Stats.OutputTokens
				if ev.Status == "success" {
					stopReason = bridle.StopReasonModelDone
				} else {
					stopReason = bridle.StopReasonError
				}
				gotResult = true
			}
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return bridle.ProviderResult{}, fmt.Errorf("antigravitycli: stream read: %w", err)
	}

	if !gotResult {
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

// buildPrompt assembles the messages into a single prompt string for the CLI.
func buildPrompt(req bridle.ProviderRequest) string {
	if len(req.Messages) == 0 {
		return ""
	}

	var contextLines []string
	var userPrompt string

	for i, m := range req.Messages {
		if m.Role == "user" && i == len(req.Messages)-1 {
			userPrompt = m.Content
		} else if m.Content != "" {
			contextLines = append(contextLines, fmt.Sprintf("[%s]: %s", m.Role, m.Content))
		}
	}

	preamble := ""
	if req.AppendSystemPrompt != "" {
		preamble = "System: " + req.AppendSystemPrompt + "\n\n"
	}

	if len(contextLines) == 0 {
		return preamble + userPrompt
	}
	return fmt.Sprintf("%sPrior context:\n%s\n\n%s", preamble, strings.Join(contextLines, "\n"), userPrompt)
}

func (p *Provider) buildCLIArgs(req bridle.ProviderRequest) []string {
	prompt := buildPrompt(req)

	args := []string{"-p", prompt, "--output-format", "stream-json"}

	if p.Yolo {
		args = append(args, "--dangerously-skip-permissions")
	}
	if req.Model != "" {
		args = append(args, "--model", req.Model)
	}
	if req.Cwd != "" {
		args = append(args, "--add-dir", req.Cwd)
	}

	allowed := p.AllowedTools
	if len(req.Tools) > 0 {
		perTurn := make([]string, 0, len(req.Tools))
		for _, t := range req.Tools {
			if t.Name != "" {
				perTurn = append(perTurn, t.Name)
			}
		}
		if len(perTurn) > 0 {
			allowed = perTurn
		}
	}
	for _, t := range allowed {
		args = append(args, "--allowed-tools", t)
	}

	if req.Session.ID != "" && !req.Session.New {
		args = append(args, "--conversation", req.Session.ID)
	}

	args = append(args, p.ExtraArgs...)
	return args
}
