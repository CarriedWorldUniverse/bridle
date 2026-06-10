// Package antigravitycli implements the bridle Provider interface using the
// Antigravity CLI (agy) in headless mode (agy -p).
//
// Category: subprocess-stream. agy runs its own agentic loop internally and
// has NO structured-output mode — `-p/--print` runs a single prompt
// non-interactively and prints the final response as PLAIN TEXT. bridle
// captures that text and emits it as one assistant-text event. There is no
// per-tool-call streaming: tool calls happen inside agy and are not observable
// here. BeforeToolCall/AfterToolCall do not fire.
//
// Auth: agy itself decides — Google login / credentials at ~/.gemini. Bridle
// does not pass credentials.
//
// Session continuity: resume uses agy's -c/--continue (most recent
// conversation). Session.ID only signals THAT a resume is wanted — its value
// is never passed to agy (it's a funnel uuid, not an agy conversation id).
package antigravitycli

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"regexp"
	"strings"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

const providerID = bridle.ProviderAntigravityCLI

var staleConversationWarningRE = regexp.MustCompile(`^Warning: conversation "[^"]+" not found\.$`)

// Provider implements bridle.Provider by shelling out to the Antigravity CLI (agy).
type Provider struct {
	// AntigravityPath is the path to the agy binary. Defaults to "agy" (PATH lookup).
	AntigravityPath string
	// Yolo passes --dangerously-skip-permissions so agy auto-approves tool calls (required for headless).
	Yolo bool
	// ExtraArgs are appended verbatim to the agy invocation.
	ExtraArgs []string
}

// New returns an antigravitycli Provider with default settings (yolo on).
// Headless agent use almost always wants yolo on so that tool calls do not block on prompts.
func New() *Provider {
	return &Provider{AntigravityPath: "agy", Yolo: true}
}

func (p *Provider) Name() bridle.ProviderID { return providerID }

func (p *Provider) Capabilities() bridle.ProviderCapabilities {
	return bridle.ProviderCapabilities{
		Category:               bridle.CategorySubprocessStream,
		SupportsCustomTools:    false,
		SupportsBeforeToolCall: false,
		SupportsAfterToolCall:  false,
		SupportsMCP:            false,
	}
}

// RunTurn invokes agy in headless print mode and emits its plain-text output as
// a single assistant-text event. Tool calls are executed by agy internally;
// bridle's ToolRunner is not called. Cancellation via ctx sends SIGTERM then
// SIGKILL after a 5s grace period.
func (p *Provider) RunTurn(ctx context.Context, req bridle.ProviderRequest, sink bridle.EventSink) (bridle.ProviderResult, error) {
	antigravityPath := p.AntigravityPath
	if antigravityPath == "" {
		antigravityPath = "agy"
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

	captureDone := make(chan struct{})
	var result bridle.ProviderResult
	var parseErr error
	go func() {
		defer close(captureDone)
		result, parseErr = capturePlainText(stdoutPipe, sink)
	}()

	waitErr := cmd.Wait()
	close(procExited) // signal the cancel watcher that the process is gone
	<-captureDone

	if waitErr != nil && parseErr == nil {
		if ctx.Err() != nil {
			result.StopReason = bridle.StopReasonAborted
		} else {
			sink.Emit(bridle.TurnError{Err: fmt.Errorf("antigravitycli: %w", waitErr), Stage: bridle.TurnErrorStageSubprocessExit})
			return bridle.ProviderResult{}, fmt.Errorf("antigravitycli: agy error: %w (stderr: %s)", waitErr, stderr.String())
		}
	}

	return result, parseErr
}

// capturePlainText reads agy's -p output (the final response as plain text) and
// emits it as one assistant-text event. agy runs its agentic loop internally,
// so there are no intermediate tool-call events to stream.
func capturePlainText(r io.Reader, sink bridle.EventSink) (bridle.ProviderResult, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return bridle.ProviderResult{}, fmt.Errorf("antigravitycli: read output: %w", err)
	}

	var (
		finalText    string
		sessionDelta []bridle.SessionEvent
	)
	text, staleConversation := stripLeadingStaleConversationWarnings(string(raw))
	if text := strings.TrimSpace(text); text != "" {
		bridle.EmitAssistantText(sink, &finalText, &sessionDelta, providerID, text)
	}
	if staleConversation {
		sessionDelta = nil
	}

	return bridle.ProviderResult{
		FinalText:    finalText,
		StopReason:   bridle.StopReasonModelDone,
		SessionDelta: sessionDelta,
	}, nil
}

func stripLeadingStaleConversationWarnings(text string) (string, bool) {
	lines := strings.SplitAfter(text, "\n")
	removed := false
	i := 0
	for i < len(lines) {
		line := strings.TrimRight(lines[i], "\r\n")
		if !staleConversationWarningRE.MatchString(line) {
			break
		}
		removed = true
		i++
	}
	return strings.Join(lines[i:], ""), removed
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

	// agy -p runs the prompt non-interactively and prints the final response.
	// agy has no --output-format or --allowed-tools; it manages its own tools.
	args := []string{"-p", prompt}

	if p.Yolo {
		args = append(args, "--dangerously-skip-permissions")
	}
	if req.Model != "" {
		args = append(args, "--model", req.Model)
	}
	if req.Cwd != "" {
		args = append(args, "--add-dir", req.Cwd)
	}
	if req.Session.ID != "" && !req.Session.New {
		// Resume via -c (continue the MOST RECENT conversation) rather than
		// --conversation <id>: the funnel's Session.ID is its own (claude-
		// shaped) uuid, never an id agy minted — and plain-text mode gives us
		// no way to learn agy's real conversation id. agy rejects unknown ids
		// ("Warning: conversation … not found") and starts fresh, silently
		// dropping cross-turn memory. In a single-agent home (one agy user
		// per pod), "most recent" IS this aspect's thread, so -c restores
		// continuity without needing the id at all.
		args = append(args, "-c")
	}

	args = append(args, p.ExtraArgs...)
	return args
}
