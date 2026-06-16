package bridle

// SystemPromptMode selects how AppendSystemPrompt is applied:
//
//   - SystemPromptAppend (default): Append to bridle's built-in base prompt,
//     like --append-system-prompt. The caller's composed system prompt
//     extends the default framing.
//
//   - SystemPromptReplace: Replace the entire base prompt with the caller's
//     text, like --system-prompt. The default framing is completely hidden;
//     only what the caller specifies appears in the turn. Useful for callers
//     that want to own the system framing entirely and opt out of bridle's
//     built-in rules (e.g., nexus.md).
//
// Zero value = SystemPromptAppend (the default / append mode). Field is
// named so that its zero value has the safer behavior — opt-in for the
// override, not opt-out.
type SystemPromptMode string

const (
	SystemPromptAppend  SystemPromptMode = "" // empty means "append" — keep the base prompt + extend it (default)
	SystemPromptReplace SystemPromptMode = "replace"
)

func (m SystemPromptMode) IsValid() bool {
	switch m {
	case SystemPromptAppend, SystemPromptReplace:
		return true
	default:
		return false
	}
}