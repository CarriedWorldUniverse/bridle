package bridle

import (
	"encoding/json"
	"errors"
)

// Event is the union type for all observable harness events.
type Event interface {
	event() // unexported marker; keeps the interface closed to this package
}

// ModelChunk carries a streamed text fragment from the model.
type ModelChunk struct {
	Text string
}

// ToolCallStart fires when the model requests a tool call, before execution.
type ToolCallStart struct {
	ID   string
	Name string
	Args json.RawMessage
}

// ToolCallResult fires after the tool runner returns (or errors).
type ToolCallResult struct {
	ID     string
	Result json.RawMessage
	Err    string // non-empty if the tool runner returned an error
}

// StepBoundary fires between tool-call rounds.
// Step 1 = the first round; fires after its results are sent back to the model.
type StepBoundary struct {
	Step int
}

// TurnDone fires after the turn completes successfully.
type TurnDone struct {
	Result TurnResult
}

// TurnError fires when the provider or harness hits a non-recoverable
// error. Never panics across the harness boundary.
//
// Stage labels the pipeline location where the error surfaced — useful
// for log routing and dashboards. See TurnErrorStage for the
// enumerated values bridle emits; consumers MAY observe other strings
// (forwarded from wire, set by tests). Free-form is intentional.
type TurnError struct {
	Err   error
	Stage TurnErrorStage
}

// TurnErrorStage names a pipeline location that produced a TurnError.
// The underlying type is a string so consumers that just log it (or
// receive forwarded values from the wire) continue to work.
type TurnErrorStage string

const (
	// TurnErrorStageHarnessRecover — panic trap inside Harness.RunTurn.
	TurnErrorStageHarnessRecover TurnErrorStage = "harness-recover"
	// TurnErrorStageProvider — provider.RunTurn returned a non-nil
	// error before producing a complete result.
	TurnErrorStageProvider TurnErrorStage = "provider"
	// TurnErrorStageRetry — a transient provider error is being
	// retried (claudecode); informational, not terminal.
	TurnErrorStageRetry TurnErrorStage = "retry"
	// TurnErrorStageProviderAPIError — claudecode stream-json reported
	// is_api_error=true; the run continues but the kind is surfaced.
	TurnErrorStageProviderAPIError TurnErrorStage = "provider_api_error"
	// TurnErrorStageSubprocessExit — a subprocess-stream provider
	// exited non-zero and the classifier had no better label.
	TurnErrorStageSubprocessExit TurnErrorStage = "subprocess_exit"
	// TurnErrorStageSubprocessExitPartial — subprocess exited non-zero
	// AFTER producing parseable assistant content; the partial result
	// is preserved with StopReason=process_exit.
	TurnErrorStageSubprocessExitPartial TurnErrorStage = "subprocess_exit_partial"
	// TurnErrorStageStderrOutput — subprocess wrote non-empty stderr
	// on a clean exit; surfaced as a warning, not a failure.
	TurnErrorStageStderrOutput TurnErrorStage = "stderr_output"
	// TurnErrorStageStreamTruncated — the provider's event stream
	// ended without a terminal result event.
	TurnErrorStageStreamTruncated TurnErrorStage = "stream_truncated"
)

// ProviderErrorKind classifies a provider-level error so callers can
// surface a distinct diagnosis string instead of an opaque exit code.
type ProviderErrorKind string

const (
	ProviderErrorAuthFailed     ProviderErrorKind = "auth_failed"
	ProviderErrorRateLimit      ProviderErrorKind = "rate_limit"
	ProviderErrorServerError    ProviderErrorKind = "server_error"
	ProviderErrorNetworkError   ProviderErrorKind = "network_error"
	ProviderErrorTimeout        ProviderErrorKind = "timeout"
	ProviderErrorTLSError       ProviderErrorKind = "tls_error"
	// ProviderErrorSubprocessExit is the fallback kind used when a
	// subprocess-style provider exited non-zero and no other
	// classification matched. Callers can filter for this via
	// IsProviderErrorKind to handle the generic-failure case
	// distinctly from the more specific classes above.
	ProviderErrorSubprocessExit ProviderErrorKind = "subprocess_exit"
)

// ProviderError is a classified provider-level error.
type ProviderError struct {
	Kind    ProviderErrorKind
	Message string
	Err     error // underlying error (may be nil)
}

func (e *ProviderError) Error() string {
	if e.Err != nil {
		return e.Message + ": " + e.Err.Error()
	}
	return e.Message
}

func (e *ProviderError) Unwrap() error { return e.Err }

// IsProviderErrorKind reports whether err (or any error in its chain) is
// a ProviderError with the given kind.
func IsProviderErrorKind(err error, kind ProviderErrorKind) bool {
	var pe *ProviderError
	if errors.As(err, &pe) {
		return pe.Kind == kind
	}
	return false
}

func (ModelChunk) event()     {}
func (ToolCallStart) event()  {}
func (ToolCallResult) event() {}
func (StepBoundary) event()   {}
func (TurnDone) event()       {}
func (TurnError) event()      {}
