package bridle

import (
	"encoding/json"
	"errors"
	"time"
)

// Event is the union type for all observable harness events.
type Event interface {
	event() // unexported marker; keeps the interface closed to this package
}

// ModelChunk carries a streamed text fragment from the model.
type ModelChunk struct {
	Text string
	TS   time.Time // stamped by the harness at emission; zero outside a harness turn
}

// ReasoningChunk carries a streamed extended-thinking/reasoning-content
// fragment from the model — the reasoning_delta half of the
// agora-spec-bridle §2 event vocab (ModelChunk covers text_delta).
// Populated by claude's ThinkingDelta branch and openai's
// reasoning_content live-emit (both providers extracted this today but
// never streamed it before NEX-767 T7).
type ReasoningChunk struct {
	Text string
	TS   time.Time // stamped by the harness at emission; zero outside a harness turn
}

// Warning fires for a non-fatal, once-per-session-worthy condition the
// harness or a provider wants to surface without failing the turn —
// e.g. an unsupported effort tier/knob falling back to a default
// (agora-spec-bridle §3: "Unsupported tier or knob → drop with a
// warning event once per session"). bridle itself is stateless and
// emits every time it hits the condition; dedup (once-per-session) is
// the consumer's job (agora's TUI, per the blueprint's open question 2).
type Warning struct {
	Kind    string
	Message string
	TS      time.Time // stamped by the harness at emission; zero outside a harness turn
}

// ToolCallStart fires when the model requests a tool call, before execution.
type ToolCallStart struct {
	ID   string
	Name string
	Args json.RawMessage
	TS   time.Time // stamped by the harness at emission; zero outside a harness turn
}

// ToolCallResult fires after the tool runner returns (or errors).
type ToolCallResult struct {
	ID     string
	Result json.RawMessage
	Err    string    // non-empty if the tool runner returned an error
	TS     time.Time // stamped by the harness at emission; zero outside a harness turn
}

// StepBoundary fires between tool-call rounds.
// Step 1 = the first round; fires after its results are sent back to the model.
type StepBoundary struct {
	Step int
	TS   time.Time // stamped by the harness at emission; zero outside a harness turn
}

// TurnDone fires after the turn completes successfully.
type TurnDone struct {
	Result TurnResult
	TS     time.Time // stamped by the harness at emission; zero outside a harness turn
}

// MCPServerFailed fires when an MCP server fails to connect/initialize
// during turn setup. NEX-596: such a failure is non-fatal — the server's
// tools are dropped and the turn proceeds with the remaining servers.
// This event surfaces the dropped server for observability.
type MCPServerFailed struct {
	Server string
	Err    error
	TS     time.Time // stamped by the harness at emission; zero outside a harness turn
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
	TS    time.Time // stamped by the harness at emission; zero outside a harness turn
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
	// TurnErrorStageResumeFallback — a session resume failed because the
	// referenced session was missing/corrupt, and the provider fell back
	// to a fresh session. Informational/warning, not terminal: the turn
	// proceeds without the prior session's context. Distinct from
	// TurnErrorStageRetry (same session, transient error).
	TurnErrorStageResumeFallback TurnErrorStage = "resume_fallback"
)

// ProviderErrorKind classifies a provider-level error so callers can
// surface a distinct diagnosis string instead of an opaque exit code.
type ProviderErrorKind string

const (
	ProviderErrorAuthFailed   ProviderErrorKind = "auth_failed"
	ProviderErrorRateLimit    ProviderErrorKind = "rate_limit"
	ProviderErrorServerError  ProviderErrorKind = "server_error"
	ProviderErrorNetworkError ProviderErrorKind = "network_error"
	ProviderErrorTimeout      ProviderErrorKind = "timeout"
	ProviderErrorTLSError     ProviderErrorKind = "tls_error"
	// ProviderErrorConfig is a non-transient setup failure: the CLI
	// binary is missing from PATH, a referenced config file/profile is
	// absent, or a required flag/argument is malformed. Retrying is
	// futile — the fix is operator configuration, not a re-run.
	ProviderErrorConfig ProviderErrorKind = "config_error"
	// ProviderErrorCrash is an abnormal subprocess termination distinct
	// from an orderly non-zero exit: a fatal signal (segfault/abort), an
	// out-of-memory kill, or a panic/stack-overflow in the CLI itself.
	// Surfaced separately so operators can tell "the model API rejected
	// us" (auth/rate) from "the CLI process itself died".
	ProviderErrorCrash ProviderErrorKind = "subprocess_crash"
	// ProviderErrorSubprocessExit is the fallback kind used when a
	// subprocess-style provider exited non-zero and no other
	// classification matched. Callers can filter for this via
	// IsProviderErrorKind to handle the generic-failure case
	// distinctly from the more specific classes above.
	ProviderErrorSubprocessExit ProviderErrorKind = "subprocess_exit"
	// ProviderErrorRefusal is a model-level safety refusal (the engine
	// declined to continue the turn) rather than a transport/auth/rate
	// failure. Distinct so callers don't retry a refusal the way they'd
	// retry a transient error — re-sending the same turn to the same
	// model is very unlikely to produce a different outcome. Surfaced by
	// claudesdk when the Agent SDK reports stop_reason "refusal" (bridle
	// spec NEX-745 §7).
	ProviderErrorRefusal ProviderErrorKind = "refusal"
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

// ErrorClass is the Stream-facing error taxonomy (agora-spec-bridle §3:
// "auth | rate_limit | overloaded | context_length | schema | network |
// refusal | provider"). agora's retry policy keys off the class:
// rate_limit/overloaded/network are retryable with backoff, auth
// surfaces immediately, context_length routes to the context manager,
// refusal is non-retryable content surfaced to the turn/approval layer,
// provider is the residual "something else went wrong" bucket.
//
// T1/T7 adds the TYPE and the mapping-table home (internal/normalize's
// ProviderErrorClass) so Stream's error{class} event has somewhere to
// land; per-lane DETECTION of overloaded/context_length/schema/refusal
// from real wire errors is T3 follow-up work — today's ProviderErrorKind
// values don't distinguish those four yet.
type ErrorClass string

const (
	ErrorClassAuth          ErrorClass = "auth"
	ErrorClassRateLimit     ErrorClass = "rate_limit"
	ErrorClassOverloaded    ErrorClass = "overloaded"
	ErrorClassContextLength ErrorClass = "context_length"
	ErrorClassSchema        ErrorClass = "schema"
	ErrorClassNetwork       ErrorClass = "network"
	ErrorClassRefusal       ErrorClass = "refusal"
	ErrorClassProvider      ErrorClass = "provider"
)

func (ModelChunk) event()      {}
func (ReasoningChunk) event()  {}
func (ToolCallStart) event()   {}
func (ToolCallResult) event()  {}
func (StepBoundary) event()    {}
func (TurnDone) event()        {}
func (TurnError) event()       {}
func (MCPServerFailed) event() {}
func (Warning) event()         {}
