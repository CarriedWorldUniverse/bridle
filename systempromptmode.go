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

	// SystemPromptFull is the agora-spec-bridle §1 vocabulary alias for
	// SystemPromptReplace ("full|append" per the registry's
	// system_prompt_mode capability field). Callers that speak agora's
	// vocabulary can set this directly on TurnRequest/ProviderRequest;
	// call Normalize() before switching on the value so "full" and
	// "replace" are treated identically everywhere.
	SystemPromptFull SystemPromptMode = "full"
)

func (m SystemPromptMode) IsValid() bool {
	switch m {
	case SystemPromptAppend, SystemPromptReplace, SystemPromptFull:
		return true
	case "append":
		// Explicit non-zero spelling of SystemPromptAppend. TurnRequest's
		// zero-value-means-append idiom doesn't help catalog data (a TOML
		// row's field can't distinguish "explicitly append" from
		// "field omitted"), so the registry's ModelCapabilities.
		// SystemPromptMode rows spell it out as "append" rather than
		// leaving it blank. Equivalent to SystemPromptAppend everywhere.
		return true
	default:
		return false
	}
}

// Normalize collapses the agora-vocabulary alias onto bridle's existing
// modes: SystemPromptFull becomes SystemPromptReplace; everything else
// passes through unchanged. Callers that switch on SystemPromptMode
// (e.g. claudecode's buildCLIArgs) should call Normalize() first so
// "full" and "replace" take the same branch without duplicating cases.
func (m SystemPromptMode) Normalize() SystemPromptMode {
	if m == SystemPromptFull {
		return SystemPromptReplace
	}
	return m
}
