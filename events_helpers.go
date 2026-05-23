package bridle

// AppendAssistantText accumulates a chunk of model-emitted text into
// *finalText and appends a matching assistant-role SessionEvent (with
// Provider set) to *sessionDelta. Does NOT emit a ModelChunk — that's
// the caller's job when the chunk is observed live during a streaming
// loop.
//
// Centralises the trio (accumulate + SessionEvent shape) that every
// direct-API provider's extractResult / subprocess-stream provider's
// text branch repeats. Concretely, this is the spot where forgetting
// to set Provider silently broke ParseSessionEvent in three providers;
// using this helper makes the field impossible to omit.
func AppendAssistantText(finalText *string, sessionDelta *[]SessionEvent, providerID ProviderID, text string) {
	*finalText += text
	*sessionDelta = append(*sessionDelta, SessionEvent{
		Provider: providerID,
		Role:     RoleAssistant,
		Content:  text,
	})
}

// EmitAssistantText is AppendAssistantText plus a live ModelChunk
// emit. Use from subprocess-stream provider parsers (claudecode,
// geminicli) where the same text is BOTH streamed live AND folded
// into the final result/session log. Direct-API providers that emit
// chunks inside their SDK stream loop and lower a separate aggregate
// in extractResult should use AppendAssistantText in the lowering
// path and call sink.Emit directly in the stream loop.
func EmitAssistantText(sink EventSink, finalText *string, sessionDelta *[]SessionEvent, providerID ProviderID, text string) {
	sink.Emit(ModelChunk{Text: text})
	AppendAssistantText(finalText, sessionDelta, providerID, text)
}
