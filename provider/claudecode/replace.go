// Package claudecode implements the bridle Provider interface using the
// claude-code CLI in headless mode (claude -p --output-format stream-json --verbose).
//
// Category: subprocess-stream. The CLI manages its own agentic loop;
// bridle parses the stdout event stream. BeforeToolCall does not fire;
// AfterToolCall fires after each parsed tool_use/tool_result pair (observe-only).
//
// Session continuity: the caller passes a SessionHandle with a non-empty ID.
// SessionHandle.New=true → the funnel is starting a fresh session for this
// ID, so we pass --session-id <id> (CLI creates the jsonl). New=false → the
// funnel is continuing an existing session, so we pass --resume <id> (CLI
// loads the jsonl, errors if not found).
package claudecode

import (
	"fmt"
	"os"
)

// replaceSystemPromptArgs returns the CLI args for req.AppendSystemPrompt when
// req.SystemPromptMode == SystemPromptReplace. It matches appendSystemPromptArgs
// but emits --system-prompt / --system-prompt-file instead of --append-*.
func replaceSystemPromptArgs(body string) ([]string, string, error) {
	if len(body) <= systemPromptSpillThresholdBytes {
		return []string{"--system-prompt", body}, "", nil
	}
	f, err := os.CreateTemp("", "bridle-sysprompt-*.txt")
	if err != nil {
		return nil, "", fmt.Errorf("claudecode: tempfile for system prompt (replace): %w", err)
	}
	if _, err := f.WriteString(body); err != nil {
		_ = f.Close()
		_ = os.Remove(f.Name())
		return nil, "", fmt.Errorf("claudecode: write system prompt tempfile (replace): %w", err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(f.Name())
		return nil, "", fmt.Errorf("claudecode: close system prompt tempfile (replace): %w", err)
	}
	return []string{"--system-prompt-file", f.Name()}, f.Name(), nil
}