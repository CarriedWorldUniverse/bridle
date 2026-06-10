package bridle

import (
	"encoding/json"
	"regexp"
	"strings"
	"time"
)

// ToolCallStrictness controls how the tool-call contract reacts to a
// detected leak or an unparseable tool call (NEX-581). It is a
// per-request / per-aspect knob the funnel sets; the harness defaults to
// the repair-then-retry behavior when the field is empty.
//
// The contract is the "feels dumb" insurance: whatever an engine emits,
// bridle delivers either a well-formed tool call OR a clean text turn.
// The strictness knob picks how hard bridle works to get there.
type ToolCallStrictness string

const (
	// ToolCallStrictnessRepairThenRetry is the default. On a detected
	// leak, bridle first attempts a structural repair (no model round
	// trip). If repair can't recover a clean result — and the turn was
	// supposed to call a tool, or the content is still garbled — bridle
	// retries the round ONCE with a tightened instruction. Only after
	// repair+retry exhaust does it surface the best-effort cleaned text,
	// flagged. Builders should use this: never ship a degraded tool call.
	ToolCallStrictnessRepairThenRetry ToolCallStrictness = "repair-then-retry"

	// ToolCallStrictnessTolerant accepts the structurally repaired result
	// without a retry round. Research aspects can use this: a cleaned
	// text turn is acceptable, and the extra round isn't worth the cost.
	ToolCallStrictnessTolerant ToolCallStrictness = "tolerant"
)

// effectiveStrictness resolves the empty/zero value to the default.
func effectiveStrictness(s ToolCallStrictness) ToolCallStrictness {
	if s == "" {
		return ToolCallStrictnessRepairThenRetry
	}
	return s
}

// retryNudge is appended to the system prompt on the single retry round
// after a leak that repair couldn't fully recover. It is a gentle,
// engine-agnostic instruction; engines that honor it produce a clean
// turn the second time.
const retryNudge = "\n\nIMPORTANT: respond with a valid tool call or plain text only. " +
	"Do not emit protocol tokens, channel markers, or tool-call grammar as literal text."

// leakPatterns is the documented, extendable set of raw protocol/reasoning
// tokens that, when found in a model's surfaced text, signal that an
// engine's parser leaked its internal grammar into the response (NEX-581).
//
// Observed classes (live gemma A/B, 2026-06-11, ollama openai-compat shim):
//   - harmony/channel control tokens: <|channel|>, <|message|>, <|start|>,
//     <|end|>, <|tool_call|>, and the malformed <tool_call|> variant the
//     shim emitted.
//   - bare harmony channel-name leakage where the <|channel|> wrapper was
//     itself stripped but the channel label survived: "assistantfinal",
//     "commentary" appearing as a control prefix.
//
// These patterns must stay SPECIFIC: they target the literal control-token
// grammar, never ordinary prose. The bare-channel patterns are anchored so
// the word "commentary" or "final" in normal text can't trip them — only
// the control-prefix forms (e.g. "<|channel|>commentary" residue, or a
// line that is exactly a channel marker) match. False-positive leak
// detection is the key risk: a too-broad pattern would mangle valid output,
// so each entry is justified against a captured leak sample.
//
// Extend this var (don't replace it) when a new engine leaks a new token
// class; add a detect-corpus test alongside.
var leakPatterns = []*regexp.Regexp{
	// Harmony <|channel|> / <|start|> control token immediately followed by
	// its channel-name label (final/commentary/analysis/thought) and an
	// optional role word. Stripping the bare <|channel|> token alone would
	// strand the channel-name label as orphaned text ("final", "commentary")
	// in the surfaced output, so this composite pattern removes the label
	// too. Must run BEFORE the bare <|word|> stripper below.
	regexp.MustCompile(`<\|(?:channel|start)\|>\s*(?:final|commentary|analysis|thought|assistant|system|user)?`),
	// Angle-bracket harmony control tokens. Covers <|channel|>, <|message|>,
	// <|start|>, <|end|>, <|tool_call|>, <|im_start|>, <|im_end|>, etc.
	// The pipe-delimited <|word|> form is never valid surfaced text.
	regexp.MustCompile(`<\|[a-zA-Z_]+\|>`),
	// The malformed <tool_call|> / <tool_call> variants the compat shim
	// emitted (missing the leading pipe). Anchored to the known tool/channel
	// token names so an HTML-ish "<something>" in prose isn't caught.
	regexp.MustCompile(`<\/?(?:tool_call|channel|message|start|end)\|?>`),
	// Bare harmony channel markers that leaked without their <|...|> wrapper:
	// the channel label fused to the word "channel" or appearing as the
	// concatenated "assistantfinal" / "assistantcommentary" forms the shim
	// produced. Anchored to the harmony vocabulary so normal use of "final"
	// or "commentary" in prose is safe.
	regexp.MustCompile(`\bassistant(?:final|commentary)\b`),
	regexp.MustCompile(`<\|?(?:channel|constrain)\|?>?\s*(?:final|commentary|analysis|thought)\b`),
}

// jsonToolCallInText matches a JSON object that looks like a tool call the
// engine emitted as LITERAL TEXT in the content instead of as a structured
// call: {"name": "...", "arguments": {...}} (or "args"/"parameters" /
// "input" as the args key). The match is captured so repair can extract it
// into a proper ToolInvocation. Kept deliberately narrow — the object must
// lead with a "name" key and carry a recognised args key — so ordinary JSON
// in the model's prose (e.g. an example, or a data payload) isn't mistaken
// for a leaked call.
var jsonToolCallInText = regexp.MustCompile(
	`\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|args|parameters|input)"\s*:\s*(\{.*?\}|\[.*?\]|"[^"]*"|null)\s*\}`,
)

// leakReport is the outcome of scanning one provider result.
type leakReport struct {
	// detected is true when any leakPattern or a JSON-tool-call-as-text
	// matched the surfaced text (or a tool-call argument).
	detected bool
	// detail is a short human-readable label of what tripped detection,
	// for the observability event.
	detail string
}

// detectLeak scans the provider's surfaced text and tool-call arguments
// for raw protocol tokens or a tool-call-as-text. A positive result is a
// contract violation that triggers repair (and possibly retry).
//
// It scans FinalText (the leaked-token class) and the JSON args of each
// ToolInvocation (a structured call whose args still carry leaked tokens).
// It does NOT flag a perfectly clean structured tool call — that's the
// happy path and must pass untouched.
func detectLeak(res ProviderResult) leakReport {
	for _, re := range leakPatterns {
		if loc := re.FindString(res.FinalText); loc != "" {
			return leakReport{detected: true, detail: "protocol-token:" + loc}
		}
	}
	// A tool call the engine emitted as literal text rather than a
	// structured call is a leak only when the model produced NO structured
	// tool call (otherwise the JSON-looking text is probably legitimate
	// prose about a call). If there are structured calls already, the
	// text-JSON isn't the intended invocation.
	if len(res.ToolCalls) == 0 {
		if jsonToolCallInText.MatchString(res.FinalText) {
			return leakReport{detected: true, detail: "tool-call-as-text"}
		}
	}
	// Leaked protocol tokens inside a structured call's arguments: the
	// engine produced a real call but its args field carries garbage.
	for _, tc := range res.ToolCalls {
		for _, re := range leakPatterns {
			if loc := re.FindString(string(tc.Args)); loc != "" {
				return leakReport{detected: true, detail: "args-token:" + loc}
			}
		}
	}
	return leakReport{detected: false}
}

// repairOutcome is the result of a structural repair attempt.
type repairOutcome struct {
	// text is the cleaned surfaced text (protocol tokens stripped).
	text string
	// toolCalls is the recovered structured call set. When repair
	// extracted a tool-call-as-text into a proper invocation, this holds
	// it; otherwise it carries the original (token-stripped) calls.
	toolCalls []ToolInvocation
	// clean is true when repair fully recovered a usable result: either
	// a structured tool call was extracted, or the remaining text is free
	// of protocol tokens. When false, the content is still garbled and the
	// caller should retry (strict) or surface flagged (tolerant).
	clean bool
	// extractedCall is true when repair pulled a tool-call-as-text out of
	// the content into a structured ToolInvocation.
	extractedCall bool
	// detail labels what repair did, for the observability event.
	detail string
}

// repairLeak attempts an engine-agnostic structural repair WITHOUT a model
// round trip:
//   - If the content carries a JSON tool-call-as-text, extract it into a
//     proper ToolInvocation and strip it from the text.
//   - Strip the leaked protocol tokens from the text.
//   - Report clean=true when the result is usable; clean=false when, after
//     stripping, the text is empty/garbled AND no structured call survived
//     (the caller then retries or surfaces flagged).
func repairLeak(res ProviderResult) repairOutcome {
	text := res.FinalText
	toolCalls := res.ToolCalls
	var extracted bool
	var detail string

	// 1. Extract a tool-call-as-text into a structured call, if present and
	//    no structured call already exists.
	if len(toolCalls) == 0 {
		if m := jsonToolCallInText.FindStringSubmatchIndex(text); m != nil {
			whole := text[m[0]:m[1]]
			name := text[m[2]:m[3]]
			args := text[m[4]:m[5]]
			if inv, ok := buildInvocationFromText(name, args); ok {
				toolCalls = []ToolInvocation{inv}
				extracted = true
				detail = "extracted-tool-call"
				// Remove the JSON blob from the surfaced text.
				text = strings.Replace(text, whole, "", 1)
			}
		}
	}

	// 2. Strip leaked protocol tokens from the surfaced text.
	stripped := stripLeakTokens(text)
	if stripped != text {
		if detail == "" {
			detail = "stripped-tokens"
		} else {
			detail += "+stripped-tokens"
		}
	}
	text = strings.TrimSpace(stripped)

	// 3. Strip leaked tokens out of any structured call's args too — a
	//    structured call whose args carry garbage is still a contract
	//    violation. We only token-strip; we never rewrite the JSON value
	//    shape, so a clean call passes through byte-identical.
	toolCalls = stripToolCallArgs(toolCalls)

	// 4. Decide clean. A structured call (extracted or surviving) makes the
	//    result usable. Otherwise the text must be non-empty and free of
	//    protocol tokens.
	clean := false
	switch {
	case extracted:
		clean = true
	case len(toolCalls) > 0:
		// A pre-existing structured call survived; clean iff its args no
		// longer carry leaked tokens.
		clean = !argsCarryLeak(toolCalls)
	default:
		// Text-only turn: clean iff stripping left real content and no
		// residual protocol token remains.
		clean = text != "" && !textCarriesLeak(text)
	}

	if detail == "" {
		detail = "noop"
	}
	return repairOutcome{
		text:          text,
		toolCalls:     toolCalls,
		clean:         clean,
		extractedCall: extracted,
		detail:        detail,
	}
}

// buildInvocationFromText turns a leaked name+args text pair into a
// ToolInvocation. The args fragment must be valid JSON for the call to be
// considered recovered; a malformed args blob returns ok=false so the
// caller falls through to retry rather than ship a broken call.
func buildInvocationFromText(name, args string) (ToolInvocation, bool) {
	raw := json.RawMessage(strings.TrimSpace(args))
	if !json.Valid(raw) {
		return ToolInvocation{}, false
	}
	return ToolInvocation{
		// The leaked text carried no call id; synthesize a stable marker
		// so downstream correlation has a non-empty id. Real engines mint
		// ids; this only fires on the repair path.
		ID:   "repaired-toolcall",
		Name: name,
		Args: raw,
	}, true
}

// stripLeakTokens removes every leaked protocol token from s. Pure
// token-removal: surrounding prose is preserved, only the control tokens
// are excised. Whitespace collapse is left to the caller's TrimSpace.
func stripLeakTokens(s string) string {
	for _, re := range leakPatterns {
		s = re.ReplaceAllString(s, "")
	}
	return s
}

// stripToolCallArgs token-strips each structured call's Args. A clean
// call's args contain no leak tokens, so ReplaceAll is a no-op and the
// RawMessage is byte-identical — clean calls pass through untouched.
func stripToolCallArgs(calls []ToolInvocation) []ToolInvocation {
	if len(calls) == 0 {
		return calls
	}
	out := make([]ToolInvocation, len(calls))
	copy(out, calls)
	for i := range out {
		cleaned := stripLeakTokens(string(out[i].Args))
		out[i].Args = json.RawMessage(cleaned)
	}
	return out
}

// textCarriesLeak reports whether s still contains a leaked protocol token.
func textCarriesLeak(s string) bool {
	for _, re := range leakPatterns {
		if re.MatchString(s) {
			return true
		}
	}
	return false
}

// argsCarryLeak reports whether any call's Args still contains a leaked token.
func argsCarryLeak(calls []ToolInvocation) bool {
	for _, tc := range calls {
		if textCarriesLeak(string(tc.Args)) {
			return true
		}
	}
	return false
}

// ToolCallRepaired is the observability event emitted when the tool-call
// contract detected a leak and acted on it (NEX-581). It mirrors the
// MCPServerFailed event pattern: lightweight, stamped via the timing sink,
// surfaced so the funnel can log how often engines misbehave.
//
// Stage is one of the toolCallStage* constants. Detail is a short label of
// what tripped detection / what repair did.
type ToolCallRepaired struct {
	// Stage names the contract action: "detected", "repaired", or "retried".
	Stage string
	// Detail is a short label (which token leaked, what repair did).
	Detail string
	TS     time.Time // stamped by the harness at emission; zero outside a harness turn
}

func (ToolCallRepaired) event() {}

const (
	// toolCallStageDetected — a leak was detected (contract violation).
	toolCallStageDetected = "detected"
	// toolCallStageRepaired — a structural repair recovered a clean result.
	toolCallStageRepaired = "repaired"
	// toolCallStageRetried — repair couldn't recover; the round was retried.
	toolCallStageRetried = "retried"
	// toolCallStageFlagged — repair+retry exhausted; surfaced flagged text.
	toolCallStageFlagged = "flagged"
)
