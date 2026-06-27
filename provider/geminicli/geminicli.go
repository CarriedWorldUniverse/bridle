// Package geminicli implements the bridle Provider interface using the
// gemini CLI in headless mode (gemini -p --output-format stream-json -y).
//
// Category: subprocess-stream. The CLI manages its own agentic loop;
// bridle parses the stdout event stream. BeforeToolCall does not fire;
// AfterToolCall fires after each parsed tool_use/tool_result pair (observe-only).
//
// Auth: the CLI itself decides — Google login (Gemini Pro/Ultra subscription
// quota), API key, or Vertex AI. Bridle does not pass credentials.
//
// Session continuity: gemini's --resume flag accepts "latest" or a numeric
// index, NOT a UUID. The Session.ID field is passed through verbatim, so the
// caller must supply one of those forms. The init event's session_id is
// recorded as a SessionEvent for the funnel's records but cannot itself be
// fed back to --resume.
package geminicli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/internal/subprocess"
)

const providerID = bridle.ProviderGeminiCLI

// Provider implements bridle.Provider by shelling out to the gemini CLI.
type Provider struct {
	// GeminiPath is the path to the gemini binary. Defaults to "gemini" (PATH lookup).
	GeminiPath string
	// AllowedTools restricts which gemini-native tools the CLI may use.
	// Maps to --allowed-tools (deprecated CLI flag but still honored).
	AllowedTools []string
	// SkipTrust passes --skip-trust so the CLI runs in untrusted folders.
	SkipTrust bool
	// Yolo passes -y so the CLI auto-approves tool calls (recommended for headless).
	Yolo bool
	// ExtraArgs are appended verbatim to the gemini invocation.
	ExtraArgs []string
}

// New returns a geminicli Provider with default settings (yolo on, skip-trust on).
// Headless agent use almost always wants both — without yolo, tool calls block;
// without skip-trust, the CLI refuses to operate outside trusted folders.
func New() *Provider {
	return &Provider{GeminiPath: "gemini", Yolo: true, SkipTrust: true}
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

// RunTurn invokes the gemini CLI and streams its output as bridle events.
// Tool calls are executed by the CLI; bridle's ToolRunner is not called.
// Cancellation via ctx sends SIGTERM then SIGKILL after a 5s grace period.
//
// Session-resume robustness (NEX-588): gemini's --resume takes "latest"
// or a numeric index; a stale/out-of-range index makes the CLI exit
// non-zero. When a resume fails because the checkpoint is missing, the
// turn degrades to a FRESH session (drop --resume, re-run) with a
// logged TurnErrorStageResumeFallback warning rather than failing. Only
// on a genuine not-found signal — auth/rate/network/crash surface
// as-is.
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	resuming := req.Session.ID != ""
	result, err := p.runTurnOnce(ctx, req, sink)
	if err == nil || !resuming || ctx.Err() != nil {
		return result, err
	}
	var pe *bridle.ProviderError
	if !errors.As(err, &pe) || !subprocess.IsResumeNotFound(pe.Error()) {
		return result, err
	}
	sink.Emit(bridle.TurnError{
		Err: fmt.Errorf("geminicli: resume of session %q failed (missing/corrupt) — falling back to a fresh session; prior context is lost: %w",
			req.Session.ID, err),
		Stage: bridle.TurnErrorStageResumeFallback,
	})
	fresh := req
	fresh.Session.ID = ""
	return p.runTurnOnce(ctx, fresh, sink)
}

func (p *Provider) runTurnOnce(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	geminiPath := p.GeminiPath
	if geminiPath == "" {
		geminiPath = "gemini"
	}

	prompt := buildPrompt(req)

	args := []string{"-p", prompt, "--output-format", "stream-json"}

	if p.Yolo {
		args = append(args, "-y")
	}
	if p.SkipTrust {
		args = append(args, "--skip-trust")
	}
	if req.Model != "" {
		args = append(args, "--model", req.Model)
	}
	// Allowed tools: per-turn list from the funnel (req.Tools.Name) takes
	// precedence; fall back to the provider-level default. The CLI owns
	// execution, so the funnel sets the *allowlist*, not the schemas.
	//
	// Guard: a non-empty req.Tools with all-empty Name fields would
	// otherwise translate to allowed=[] and emit no --allowed-tools
	// flags, letting the CLI run with the full default allowlist. That's
	// a footgun (silent privilege escalation), so on degenerate input
	// we fall back to p.AllowedTools rather than the empty path.
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
	// gemini --resume accepts "latest" or a numeric index — not a UUID.
	// Pass through whatever the caller supplied; document mismatch elsewhere.
	if req.Session.ID != "" {
		args = append(args, "--resume", req.Session.ID)
	}

	args = append(args, p.ExtraArgs...)

	cmd := exec.Command(geminiPath, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("geminicli: pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("geminicli: start: %w", err)
	}

	// Cancel watcher: SIGTERM + grace period + SIGKILL. procExited is
	// closed AFTER cmd.Wait() returns; see subprocess.WatchCancel for
	// the full contract.
	procExited := make(chan struct{})
	go subprocess.WatchCancel(ctx, cmd, procExited, subprocess.TermSignal())

	streamDone := make(chan struct{})
	var result bridle.ProviderResult
	var parseErr error
	go func() {
		defer close(streamDone)
		result, parseErr = parseStream(stdoutPipe, sink)
	}()

	// Drain stdout to EOF BEFORE cmd.Wait(). StdoutPipe's contract is
	// that Wait closes the pipe FD on process exit; reaping while the
	// scan goroutine is still reading races the close and can surface a
	// spurious "file already closed" or, on a fast exit, drop unread
	// lines (including the terminal result event).
	<-streamDone
	waitErr := cmd.Wait()
	close(procExited) // signal the cancel watcher that the process is gone

	if waitErr != nil && parseErr == nil {
		if ctx.Err() != nil {
			result.StopReason = bridle.StopReasonAborted
		} else {
			// Classify the failure into an actionable kind (auth/rate/
			// network/config/crash) so the funnel logs WHAT to do rather
			// than a bare exit code (NEX-588). geminicli has no
			// CLI-specific pattern table; it relies entirely on the
			// shared classes.
			pe := classifyProviderError(stderr.String(), waitErr)
			sink.Emit(bridle.TurnError{Err: pe, Stage: bridle.TurnErrorStage(pe.Kind)})
			return bridle.ProviderResult{}, pe
		}
	}

	if parseErr == nil && result.StopReason == "" && ctx.Err() == nil {
		parseErr = fmt.Errorf("geminicli: stream ended without result event")
		sink.Emit(bridle.TurnError{Err: parseErr, Stage: bridle.TurnErrorStageStreamTruncated})
	}

	return result, parseErr
}

// parseStream reads stream-json lines and maps them to bridle events + result.
//
// Event shapes (observed from gemini CLI v0.x stream-json output):
//
//	{"type":"init", "session_id":"<uuid>", "model":"<id>"}
//	{"type":"message", "role":"user|assistant", "content":"...", "delta":true?}
//	{"type":"tool_use", "tool_name":"...", "tool_id":"...", "parameters":{...}}
//	{"type":"tool_result", "tool_id":"...", "status":"success|..."}
//	{"type":"result", "status":"success|...", "stats":{"input_tokens":..,"output_tokens":..,"tool_calls":..}}
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

	perLine := func(line []byte) {
		if line[0] != '{' {
			// CLI prints free-form banners ("YOLO mode is enabled.", etc.) on stdout.
			// Skip anything that isn't a JSON object.
			return
		}

		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(line, &head); err != nil {
			return
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
					// NB: unlike claudecode (which resets finalText on
					// every tool_use to drop draft text), this provider
					// accumulates all assistant text. If gemini-cli is
					// ever observed producing the draft → tool → rewrite
					// pattern (see bridle.ProviderResult.FinalText doc),
					// mirror claudecode's reset before the helper call.
					bridle.EmitAssistantText(sink, &finalText, &sessionDelta, providerID, ev.Content)
				}
				// User echoes are recorded but not re-emitted; the funnel already has them.
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

	if err := subprocess.ScanJSONLines(r, perLine); err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("geminicli: stream read: %w", err)
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

// classifyProviderError maps gemini's stderr to an actionable
// ProviderError kind via the shared cross-provider table (NEX-588).
// gemini-cli emits no consistent CLI-worded error strings of its own,
// so there is no provider-specific pattern list — the shared classes
// (auth/rate/network/config/crash) carry it. Falls back to a generic
// subprocess-exit kind when nothing matches.
func classifyProviderError(stderr string, waitErr error) *bridle.ProviderError {
	if kind, msg, ok := subprocess.ClassifyWithFallback(stderr, "geminicli", nil); ok {
		return &bridle.ProviderError{Kind: kind, Message: msg, Err: waitErr}
	}
	return &bridle.ProviderError{
		Kind:    bridle.ProviderErrorSubprocessExit,
		Message: "geminicli: subprocess exited with error (stderr: " + strings.TrimSpace(stderr) + ")",
		Err:     waitErr,
	}
}

// buildPrompt assembles the messages into a single prompt string for the CLI.
// Mirrors claudecode.buildPrompt — last user message becomes the prompt, prior
// turns are folded into a "Prior context:" preamble.
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
