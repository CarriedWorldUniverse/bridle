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

// ProviderErrorClass maps a bridle ProviderErrorKind onto the 8-value
// Stream error-class vocabulary (agora-spec-bridle §3, NEX-767 T7). This
// is the flagged wire→canonical mapping-table home per the blueprint
// (§8): the four classes bridle's ProviderErrorKind enum doesn't
// distinguish today — overloaded, context_length, schema, refusal — are
// per-lane DETECTION work for T3 (a future ProviderErrorKind const, or a
// dedicated raw-wire check, feeds this table a new case). Until then
// they fall through the default the same as any other unclassified
// kind.
//
// NOTE: bridle's root package (stream.go) cannot call this function —
// this package already imports the root bridle package (for the
// StopReason mappings above), so the root package importing back would
// be a cycle. Stream's own error{class} derivation therefore duplicates
// this small switch locally (see stream.go's classifyStreamError).
// Provider packages (which already import normalize) can call this
// directly once T3 wires per-lane detection.
func ProviderErrorClass(kind bridle.ProviderErrorKind) bridle.ErrorClass {
	switch kind {
	case bridle.ProviderErrorAuthFailed:
		return bridle.ErrorClassAuth
	case bridle.ProviderErrorRateLimit:
		return bridle.ErrorClassRateLimit
	case bridle.ProviderErrorNetworkError, bridle.ProviderErrorTimeout, bridle.ProviderErrorTLSError:
		return bridle.ErrorClassNetwork
	case bridle.ProviderErrorServerError, bridle.ProviderErrorCrash, bridle.ProviderErrorSubprocessExit, bridle.ProviderErrorConfig:
		return bridle.ErrorClassProvider
	default:
		return bridle.ErrorClassProvider
	}
}
