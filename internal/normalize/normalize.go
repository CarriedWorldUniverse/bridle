// Package normalize provides helpers for mapping provider-specific wire
// values to bridle's canonical StopReason values.
package normalize

import bridle "github.com/CarriedWorldUniverse/bridle"

// ClaudeStopReason maps Claude API stop_reason strings to bridle StopReason values.
func ClaudeStopReason(raw string) bridle.StopReason {
	switch raw {
	case "end_turn":
		return bridle.StopReasonModelDone
	case "max_tokens":
		return bridle.StopReasonMaxSteps
	case "tool_use":
		// tool_use is not terminal in bridle; the caller manages the loop.
		return bridle.StopReasonModelDone
	default:
		return bridle.StopReasonModelDone
	}
}

// OpenAIStopReason maps OpenAI finish_reason strings to bridle StopReason values.
func OpenAIStopReason(raw string) bridle.StopReason {
	switch raw {
	case "stop":
		return bridle.StopReasonModelDone
	case "length":
		return bridle.StopReasonMaxSteps
	case "tool_calls", "function_call":
		return bridle.StopReasonModelDone
	default:
		return bridle.StopReasonModelDone
	}
}

// GeminiStopReason maps Gemini FinishReason values to bridle StopReason values.
func GeminiStopReason(raw string) bridle.StopReason {
	switch raw {
	case "STOP", "FINISH_REASON_STOP":
		return bridle.StopReasonModelDone
	case "MAX_TOKENS":
		return bridle.StopReasonMaxSteps
	default:
		return bridle.StopReasonModelDone
	}
}

// BedrockStopReason maps AWS Bedrock Converse stop_reason strings to bridle StopReason values.
func BedrockStopReason(raw string) bridle.StopReason {
	switch raw {
	case "end_turn", "stop_sequence":
		return bridle.StopReasonModelDone
	case "max_tokens":
		return bridle.StopReasonMaxSteps
	case "tool_use":
		// non-terminal in bridle — caller manages the tool loop
		return bridle.StopReasonModelDone
	case "guardrail_intervened", "content_filtered":
		return bridle.StopReasonError
	default:
		return bridle.StopReasonModelDone
	}
}

// OllamaStopReason maps Ollama done_reason strings to bridle StopReason values.
func OllamaStopReason(raw string) bridle.StopReason {
	switch raw {
	case "stop":
		return bridle.StopReasonModelDone
	case "length":
		return bridle.StopReasonMaxSteps
	default:
		return bridle.StopReasonModelDone
	}
}
