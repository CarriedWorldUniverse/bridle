package bridle

import (
	"strings"
	"unicode"
)

// usage.go implements the usage contract (NEX-581): every completed
// turn carries input/output token counts in bridle.Usage, regardless
// of engine. A provider either reports real usage, or — when it
// returns zero (the engine didn't report and an explicit
// include-usage request didn't help) — bridle estimates the counts
// from a lightweight tokenizer and flags them Estimated. The estimate
// is a FLOOR: it is never zero for a turn that had content, it is
// always flagged, and it never hard-fails the turn (the spec's
// resolved open question: estimate + flag, never non-conformant-fail).

// normalizeUsage fills any zero token count in u by estimating it from
// the round's prompt and response text, flagging the result Estimated
// when it does. It is engine-agnostic: providers that already report
// real usage pass through untouched (Estimated stays false); providers
// that returned zero get a plausible, flagged floor.
//
// The two sides are filled independently — an engine that reports
// input but not output gets only its output estimated, and the
// reported input is preserved exactly. A genuinely empty turn (no
// prompt, no response) has nothing to estimate from and stays zero
// without being flagged: the contract is "never silently zero a turn
// that had content", not "invent tokens for an empty turn".
//
// prompt is the text the model received this round (the assembled
// request) and response is what it produced (FinalText). Callers in
// run.go pass the round's request/response so the estimate tracks the
// actual exchange.
func normalizeUsage(u Usage, prompt, response string) Usage {
	if u.InputTokens == 0 {
		if est := estimateTokens(prompt); est > 0 {
			u.InputTokens = est
			u.Estimated = true
		}
	}
	if u.OutputTokens == 0 {
		if est := estimateTokens(response); est > 0 {
			u.OutputTokens = est
			u.Estimated = true
		}
	}
	return u
}

// estimateTokens approximates the token count of s. It is deliberately
// dependency-free and approximate — a floor for cost visibility, not a
// billing-grade count. We deliberately avoid a heavy native tokenizer
// dep (tiktoken/BPE bindings) for a last-resort estimate.
//
// The heuristic blends two signals that bound real BPE tokenization
// for typical English/code text:
//   - chars/4: the well-known rough average of ~4 characters per token
//     for English prose (OpenAI's own published rule of thumb).
//   - word count: each whitespace-delimited word is at least one token;
//     this floors short, punctuation-light text the chars/4 rule would
//     under-count and keeps the estimate from collapsing on terse input.
//
// We take the max of the two so the estimate is a conservative floor in
// both regimes. The function is pure (deterministic) and monotonic with
// text length — appending text never lowers the estimate. Empty string
// returns 0 (an empty turn has no tokens to floor).
func estimateTokens(s string) int {
	if s == "" {
		return 0
	}
	chars := len([]rune(s))
	byChars := chars / 4
	byWords := wordCount(s)

	est := byWords
	if byChars > est {
		est = byChars
	}
	// Any non-empty string is at least one token (e.g. a single short
	// word, or a run of punctuation with no word breaks).
	if est < 1 {
		est = 1
	}
	return est
}

// wordCount returns the number of whitespace-delimited fields in s,
// using a unicode-aware split so multi-byte text is handled.
func wordCount(s string) int {
	return len(strings.FieldsFunc(s, unicode.IsSpace))
}
