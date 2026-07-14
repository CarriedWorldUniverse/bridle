// Package extractor defines the fact-extraction seam of ctxmap: proposal
// types and the pair-verdict vocabulary. The llama.cpp-backed implementation
// (Extractor) is behind the ctxmap_llama build tag — without it, hosts supply
// their own Proposer/PairJudge or run map-off.
package extractor

// FactProposal is what an extractor emits. It is NOT a stored fact:
// provenance is attached by the caller (which knows the transcript span),
// and status starts at PROPOSED regardless of Confidence.
type FactProposal struct {
	Statement  string   `json:"statement"`
	Kind       string   `json:"kind"`   // OBSERVED | DERIVED | PREFERENCE | CONSTRAINT
	Source     string   `json:"source"` // "user" | "assistant" — who asserted it (proposed; caller must ground-check)
	Entities   []string `json:"entities"`
	Confidence float64  `json:"confidence"`
	// Force is the utterance force behind the fact, judged by the second
	// pass: DECISION (performative — saying makes it so), DIRECTIVE (a
	// request for work; the fact is the intent), REPORT (a description of
	// world state — could be wrong), QUESTION (asserts nothing).
	Force string `json:"force,omitempty"`
}

// Utterance-force values (see FactProposal.Force).
const (
	ForceDecision  = "DECISION"
	ForceDirective = "DIRECTIVE"
	ForceReport    = "REPORT"
	ForceQuestion  = "QUESTION"
)

// Turn is one user+assistant exchange, extractor-visible text only
// (dialogue + model-authored text; tool results are excluded).
type Turn struct {
	User      string
	Assistant string
}

// ToolProposer extracts durable facts from a tool RESULT rather than dialogue.
// It is the WITHIN-TURN path for agentic coding: in a long tool loop the
// load-bearing facts (a spec's exact rules, a function signature, a constant,
// a file's role) arrive in tool output, not in user/assistant text — and they
// must be captured mid-turn so they survive after that output scrolls out of
// the model's raw window. The dialogue Propose path deliberately excludes tool
// results; this is the deliberate, separate seam for them. A Proposer may also
// implement this; the engine feature-detects it.
type ToolProposer interface {
	ProposeFromTool(focus, toolName, text string) ([]FactProposal, error)
}

// PairVerdict is the reconciler's judgment of two same-topic statements.
type PairVerdict string

const (
	PairSame        PairVerdict = "SAME"
	PairContradicts PairVerdict = "CONTRADICTS"
	PairDistinct    PairVerdict = "DISTINCT"
)
