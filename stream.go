package bridle

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// MessageRole is the abstract role a Request message carries — the
// three-role "authority gradient" agora composes with (system >
// developer > user; agora-spec-prompt §1a). bridle maps these onto each
// provider's wire shape (agora-spec-bridle §3): native developer role
// where the API has one; folded into a post-core system block on
// Anthropic-shaped APIs otherwise. T1/T7 folds BOTH system and developer
// into one system-prompt block unconditionally (see lowerRoleMessages);
// per-provider native-developer-role mapping is T5 follow-up work
// (openai.go only, per the blueprint's ticket slot-in).
type MessageRole string

const (
	MessageRoleSystem    MessageRole = "system"
	MessageRoleDeveloper MessageRole = "developer"
	MessageRoleUser      MessageRole = "user"
)

// RoleMessage is one message in a Stream Request, tagged with its
// abstract role. Ordering within a role is preserved (agora-spec-bridle §3).
type RoleMessage struct {
	Role    MessageRole
	Content string
}

// JSONSchema constrains a Stream response to a schema (agora-spec-bridle
// §3, T4 structured-output forcing). T1/T7 threads the TYPE through
// Request so the wire shape exists; it does not yet enforce anything —
// no provider is forced into structured-output mode by this facade
// today. That enforcement (native json-schema mode where capable, else
// forced single-tool-call-and-unwrap, guaranteeing validate-or-
// error{class:schema}) is T4 follow-up work.
type JSONSchema struct {
	Name        string
	Description string
	Schema      json.RawMessage
	Strict      bool
}

// Request is the agora-facing turn input for Registry.Stream
// (agora-spec-bridle §2): "{messages, tools, effort, max_tokens,
// structured, cache_hints}". bridle sees only the FINAL messages+tools
// for this request — history assembly, compaction, and session
// persistence are agora-side (agora-spec-bridle §4 non-requirements).
type Request struct {
	Messages []RoleMessage
	Tools    []ToolDef

	// Effort is the agora reasoning-effort ladder value: low | medium |
	// high | xhigh | max (agora-spec-bridle §3). T1/T7 carries the field
	// but does NOT yet translate it to any provider's reasoning params —
	// per-model effort translation is T2 follow-up work. Empty is
	// equivalent to unset (provider/model default).
	Effort string

	MaxTokens int

	// Structured requests schema-constrained output. T1/T7 threads the
	// field through but does not enforce it (see JSONSchema doc) — T4
	// follow-up work.
	Structured *JSONSchema

	// CacheHints marks the stable prefix (system + tools + skills
	// catalog) for provider-side prompt-cache control (agora-spec-bridle
	// §3). T1/T7 threads the field through but does not yet apply any
	// provider cache-control wiring from it — follow-up work.
	CacheHints []string

	// ProviderEnv is per-call auth/routing config, same contract as
	// TurnRequest.ProviderEnv / ProviderRequest.ProviderEnv.
	ProviderEnv map[string]string
}

// StreamStopReason is the Stream done{} event's terminal-signal
// vocabulary (agora-spec-bridle §2: "done{stop_reason: end|tool_calls|
// max_tokens|refusal}"). tool_calls is a NEW terminal signal Stream
// introduces: bridle's own StopReason enum treats a tool_use round as
// StopReasonModelDone (non-terminal — RunTurn's loop manages it), so
// Stream derives StreamStopToolCalls from the round's ToolCalls being
// non-empty rather than from the underlying StopReason value.
type StreamStopReason string

const (
	StreamStopEnd       StreamStopReason = "end"
	StreamStopToolCalls StreamStopReason = "tool_calls"
	StreamStopMaxTokens StreamStopReason = "max_tokens"
	StreamStopRefusal   StreamStopReason = "refusal"
)

// StreamUsage is the Stream usage{} event payload (agora-spec-bridle §2:
// "usage {input, output, cached, reasoning}").
type StreamUsage struct {
	Input     int
	Output    int
	Cached    int
	Reasoning int
}

// StreamEvent is the closed union of events Registry.Stream emits on
// its returned channel — the normalized vocabulary agora-spec-bridle §2
// specifies so agora never sees provider wire formats. Mirrors bridle's
// own Event interface pattern (events.go).
type StreamEvent interface {
	streamEvent() // unexported marker; keeps the interface closed to this package
}

// TextDeltaEvent is Stream's text_delta {s} event.
type TextDeltaEvent struct{ Text string }

// ReasoningDeltaEvent is Stream's reasoning_delta {s} event.
type ReasoningDeltaEvent struct{ Text string }

// ToolCallEvent is Stream's tool_call {id, name, args_json} event —
// always a COMPLETE call (agora-spec-bridle §2: "bridle assembles
// streamed arg fragments"). Parallel calls are emitted in the order
// ProviderResult.ToolCalls records them, which is the order the
// provider returned them in.
type ToolCallEvent struct {
	ID       string
	Name     string
	ArgsJSON json.RawMessage
}

// UsageEvent is Stream's usage {input, output, cached, reasoning} event —
// final, per request.
type UsageEvent struct{ Usage StreamUsage }

// DoneEvent is Stream's done {stop_reason} event — the terminal,
// successful-completion signal.
type DoneEvent struct{ StopReason StreamStopReason }

// ErrorEvent is Stream's error {class} event — the terminal,
// failed-completion signal.
type ErrorEvent struct {
	Class ErrorClass
	Err   error
}

// WarningEvent surfaces a bridle.Warning on the Stream channel (not part
// of agora-spec-bridle §2's core vocabulary list, but Warning events
// fire mid-stream per §3's effort-translation fallback and would
// otherwise be silently dropped for Stream consumers).
type WarningEvent struct {
	Kind    string
	Message string
}

func (TextDeltaEvent) streamEvent()      {}
func (ReasoningDeltaEvent) streamEvent() {}
func (ToolCallEvent) streamEvent()       {}
func (UsageEvent) streamEvent()          {}
func (DoneEvent) streamEvent()           {}
func (ErrorEvent) streamEvent()          {}
func (WarningEvent) streamEvent()        {}

// streamEventBuffer sizes the channel Stream returns. Buffered so a
// provider's live-emit goroutine (stampSink.Emit) never blocks on a
// slow consumer for ordinary chunk volumes; large responses still drain
// correctly because the producer goroutine keeps writing and the
// consumer keeps reading concurrently — this is just slack, not a hard
// cap on stream length.
const streamEventBuffer = 64

// Stream drives one agora turn-engine request against handle's bound
// lane and returns a channel of normalized StreamEvents (agora-spec-
// bridle §2). Dispatch is by ProviderCategory: subprocess-stream lanes
// (claude-code) call the existing Harness.RunTurn unmodified (already
// single-shot for self-executing providers); direct-api lanes call the
// new additive Harness.RunStep (single round, no tool loop — agora owns
// tool execution). Cancellation via ctx aborts the upstream request
// (both RunTurn and RunStep respect ctx through the provider's own
// RunTurn call).
//
// The channel is closed when the terminal event (done or error) has
// been sent — callers should range over it until closed rather than
// watching for a specific terminal event type.
func (r *Registry) Stream(ctx context.Context, handle ModelHandle, req Request) (<-chan StreamEvent, error) {
	r.mu.RLock()
	h, ok := r.lanes[handle.Lane]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("bridle: Stream: lane %q not bound (call Bind first)", handle.Lane)
	}

	ch := make(chan StreamEvent, streamEventBuffer)
	sink := &streamSink{ch: ch}

	go func() {
		defer close(ch)
		// This goroutine's blast radius is process-wide: Stream is the
		// seam every model call routes through, so an unrecovered panic
		// here (a bad Bind, a nil provider, any provider bug — including
		// one in h.provider.Capabilities() below, which runs before
		// either dispatch path's own recover boundary) would terminate
		// the whole process, killing every in-flight turn across every
		// lane, not just this one. Degrade to an ErrorEvent instead.
		defer func() {
			if r := recover(); r != nil {
				ch <- ErrorEvent{Class: ErrorClassProvider, Err: panicErr(r)}
			}
		}()

		caps := h.provider.Capabilities()
		if caps.Category == CategorySubprocessStream {
			streamSubprocess(ctx, h, handle, req, sink)
			return
		}
		streamDirect(ctx, h, handle, req, sink)
	}()

	return ch, nil
}

// streamDirect runs the direct-api path: RunStep (single round, no tool
// loop) against a ProviderRequest built straight from Request — agora's
// Request already carries the full messages+tools for this call, so
// there's no TurnRequest/SessionTail/UserMessage split to do (see
// agora-spec-bridle §4: bridle sees only the final messages+tools per
// request).
func streamDirect(ctx context.Context, h *Harness, handle ModelHandle, req Request, sink *streamSink) {
	preq := lowerStreamToProviderRequest(handle, req)
	presult, err := h.RunStep(ctx, preq, sink)
	emitTerminal(sink, presult, err)
}

// streamSubprocess runs the subprocess-stream path: the existing
// Harness.RunTurn, unmodified, against a TurnRequest built from Request.
// RunTurn is already single-shot for self-executing providers (run.go's
// `if !caps.SupportsCustomTools` break) — the provider's own subprocess
// owns the tool loop, and ToolCalls on the result is a record of work
// already done, not a request bridle should re-execute.
func streamSubprocess(ctx context.Context, h *Harness, handle ModelHandle, req Request, sink *streamSink) {
	treq := lowerStreamToTurnRequest(handle, req)
	result, err := h.RunTurn(ctx, treq, noopToolRunner{}, sink)
	emitTerminal(sink, ProviderResult{
		ToolCalls:  result.ToolCalls,
		Usage:      result.Usage,
		StopReason: result.StopReason,
	}, err)
}

// noopToolRunner satisfies ToolRunner for the subprocess-stream Stream
// path. Self-executing providers (claude-code) never call runner.Run —
// run.go's tool-execution block only runs when
// caps.SupportsCustomTools is true, which is false for this category —
// so this should never actually be invoked; it errors loudly rather
// than silently no-oping if that assumption ever breaks.
type noopToolRunner struct{}

func (noopToolRunner) Run(ctx context.Context, call ToolCall) (json.RawMessage, error) {
	return nil, errors.New("bridle: Stream: subprocess-stream lane invoked ToolRunner unexpectedly (self-executing providers should never call runner.Run)")
}

// lowerStreamToProviderRequest builds a ProviderRequest directly from a
// Stream Request for the direct-api (RunStep) path. System and
// developer role messages fold into one AppendSystemPrompt block,
// preserving order; user messages become "user" ProviderMessages,
// preserving order. Native developer-role mapping per provider is T5
// follow-up (openai.go only).
func lowerStreamToProviderRequest(handle ModelHandle, req Request) ProviderRequest {
	appendSystemPrompt, messages := lowerRoleMessages(req.Messages)
	return ProviderRequest{
		AppendSystemPrompt: appendSystemPrompt,
		Messages:           messages,
		Tools:              req.Tools,
		Model:              handle.Model,
		MaxOutputTokens:    req.MaxTokens,
		ProviderEnv:        req.ProviderEnv,
		// Effort, Structured, CacheHints: threaded onto Request but not
		// yet lowered into ProviderRequest fields — T2/T4/prompt-caching
		// follow-up work respectively (see their doc comments on Request).
	}
}

// lowerStreamToTurnRequest builds a TurnRequest from a Stream Request
// for the subprocess-stream (RunTurn) path. Subprocess providers like
// claude-code only look at the latest user prompt (they manage their
// own multi-turn history via --resume session state), so all user-role
// messages are concatenated into UserMessage rather than split across
// SessionTail.
func lowerStreamToTurnRequest(handle ModelHandle, req Request) TurnRequest {
	appendSystemPrompt, messages := lowerRoleMessages(req.Messages)
	var userMessage string
	for i, m := range messages {
		if i > 0 {
			userMessage += "\n\n"
		}
		userMessage += m.Content
	}
	return TurnRequest{
		AppendSystemPrompt: appendSystemPrompt,
		UserMessage:        userMessage,
		Tools:              req.Tools,
		Provider:           handle.Provider,
		Model:              handle.Model,
		MaxOutputTokens:    req.MaxTokens,
		ProviderEnv:        req.ProviderEnv,
		MaxSteps:           1, // self-executing providers already stop after one round; defensive only.
	}
}

// lowerRoleMessages folds a Request's system/developer/user messages
// into (appendSystemPrompt, userMessages): system and developer content
// concatenate in order into one system-prompt block; user messages
// become ProviderMessages in order. Shared by both Stream dispatch
// paths.
func lowerRoleMessages(msgs []RoleMessage) (appendSystemPrompt string, userMessages []ProviderMessage) {
	for _, m := range msgs {
		switch m.Role {
		case MessageRoleSystem, MessageRoleDeveloper:
			if appendSystemPrompt != "" {
				appendSystemPrompt += "\n\n"
			}
			appendSystemPrompt += m.Content
		default: // MessageRoleUser and anything unrecognized
			userMessages = append(userMessages, ProviderMessage{Role: "user", Content: m.Content})
		}
	}
	return appendSystemPrompt, userMessages
}

// streamSink implements EventSink, translating live bridle.Event
// arrivals into StreamEvents as Emit() fires. Only text_delta and
// reasoning_delta (and Warning) translate LIVE — tool calls, usage, and
// the terminal done/error are synthesized once, from the completed
// ProviderResult, by emitTerminal. This avoids double-emitting tool
// calls for subprocess-stream providers that also fire live
// ToolCallStart events during their own internal loop (see run.go's
// executeToolCall / the fake SubprocessProvider precedent) — the
// terminal ToolCalls list is already fully assembled and ordered, so
// Stream's tool_call events always come from there, uniformly across
// both dispatch paths.
type streamSink struct {
	ch chan<- StreamEvent
}

func (s *streamSink) Emit(ev Event) {
	switch e := ev.(type) {
	case ModelChunk:
		s.ch <- TextDeltaEvent{Text: e.Text}
	case ReasoningChunk:
		s.ch <- ReasoningDeltaEvent{Text: e.Text}
	case Warning:
		s.ch <- WarningEvent{Kind: e.Kind, Message: e.Message}
	}
	// ToolCallStart / ToolCallResult / StepBoundary / TurnDone / TurnError /
	// MCPServerFailed / ToolCallRepaired / ContextBudgetWarning: not part
	// of the agora Stream vocab — agora executes tools and owns
	// step/loop semantics itself. Intentionally not translated here.
}

// emitTerminal synthesizes the terminal tool_call×N + usage + done/error
// StreamEvents from a completed ProviderResult (agora-spec-bridle §2).
// An error (from RunStep/RunTurn returning non-nil) short-circuits
// straight to a single ErrorEvent — no tool_call/usage/done events fire
// for a failed round.
func emitTerminal(sink *streamSink, pr ProviderResult, err error) {
	if err != nil {
		sink.ch <- ErrorEvent{Class: classifyStreamError(err), Err: err}
		return
	}
	for _, tc := range pr.ToolCalls {
		sink.ch <- ToolCallEvent{ID: tc.ID, Name: tc.Name, ArgsJSON: tc.Args}
	}
	sink.ch <- UsageEvent{Usage: StreamUsage{
		Input:     pr.Usage.InputTokens,
		Output:    pr.Usage.OutputTokens,
		Cached:    pr.Usage.CacheReadInputTokens,
		Reasoning: pr.Usage.ReasoningTokens,
	}}
	sink.ch <- DoneEvent{StopReason: streamStopReason(pr.StopReason, len(pr.ToolCalls) > 0)}
}

// streamStopReason maps a bridle StopReason plus whether the round
// produced tool calls onto the Stream done{} vocabulary (agora-spec-
// bridle §2). tool_calls takes priority: a round with tool calls is
// ALWAYS reported as stop_reason:tool_calls regardless of the
// underlying StopReason value, since bridle's own StopReasonModelDone
// doesn't distinguish "the model finished talking" from "the model
// finished talking with a tool_use block" (RunTurn's loop manages that
// distinction internally; Stream callers need it surfaced explicitly).
//
// StopReasonRefusal doesn't exist as REAL wire detection anywhere yet
// (no provider's stop-reason mapping produces it — that's T3 per-lane
// classifier work); it exists as a vocabulary value a provider CAN set
// today (bridle.StopReason is just a string type) so this mapping and
// Stream's done{refusal} path are exercised and testable now, ahead of
// real detection landing.
func streamStopReason(sr StopReason, hasToolCalls bool) StreamStopReason {
	if hasToolCalls {
		return StreamStopToolCalls
	}
	switch sr {
	case StopReasonMaxSteps:
		return StreamStopMaxTokens
	case StopReasonRefusal:
		return StreamStopRefusal
	default:
		return StreamStopEnd
	}
}

// classifyStreamError derives an ErrorClass from a RunStep/RunTurn
// error for Stream's error{class} event. Mirrors (but cannot call —
// see internal/normalize.ProviderErrorClass's doc comment for the
// import-cycle reason) normalize.ProviderErrorClass's *ProviderError
// switch. Non-ProviderError errors (context cancellation, a panic
// converted to error, etc.) fall through to ErrorClassProvider — the
// residual bucket.
func classifyStreamError(err error) ErrorClass {
	var pe *ProviderError
	if errors.As(err, &pe) {
		switch pe.Kind {
		case ProviderErrorAuthFailed:
			return ErrorClassAuth
		case ProviderErrorRateLimit:
			return ErrorClassRateLimit
		case ProviderErrorNetworkError, ProviderErrorTimeout, ProviderErrorTLSError:
			return ErrorClassNetwork
		}
	}
	return ErrorClassProvider
}
