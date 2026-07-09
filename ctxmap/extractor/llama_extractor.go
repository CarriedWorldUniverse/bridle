//go:build ctxmap_llama

// Llama-backed extractor implementation (Qwen hybrid: 1.7B extract + 4B
// kind/source/pair judgment). Requires vendored llama.cpp libs — see Makefile.
package extractor

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/CarriedWorldUniverse/bridle/ctxmap/extractor/llama"
)

type Config struct {
	ExtractModelPath string // Qwen3-1.7B-Q8 (default extraction model)
	KindModelPath    string // Qwen3-4B-Q8; empty = reuse extraction model
	Threads          int
}

type Extractor struct {
	extract *llama.Model
	kind    *llama.Model
	threads int
}

func New(cfg Config) (*Extractor, error) {
	if cfg.Threads == 0 {
		cfg.Threads = 8
	}
	em, err := llama.LoadModel(cfg.ExtractModelPath)
	if err != nil {
		return nil, fmt.Errorf("extract model: %w", err)
	}
	km := em
	if cfg.KindModelPath != "" && cfg.KindModelPath != cfg.ExtractModelPath {
		km, err = llama.LoadModel(cfg.KindModelPath)
		if err != nil {
			return nil, fmt.Errorf("kind model: %w", err)
		}
	}
	ex := &Extractor{extract: em, kind: km, threads: cfg.Threads}
	// warmup: the first real extraction after load intermittently produced
	// nothing (a cold-model first-call effect — cost turn 1's facts in the
	// first native-driving session). A throwaway pass settles both models.
	_, _ = ex.Propose(Turn{User: "warmup: the cache size is 64.", Assistant: "Noted."}, nil, nil)
	return ex, nil
}

// Propose extracts durable facts from the current turn. context carries the
// previous K turns (K=2 per spec) and glossary the store's known entity slugs.
func (e *Extractor) Propose(current Turn, context []Turn, glossary map[string]string) ([]FactProposal, error) {
	ctx, err := e.extract.NewContext(4096, e.threads)
	if err != nil {
		return nil, err
	}
	defer ctx.Free()

	out, _, err := ctx.Generate(buildExtractionPrompt(current, context, glossary), factGrammar, 768)
	if err != nil {
		return nil, err
	}
	var facts []FactProposal
	if err := json.Unmarshal([]byte(out), &facts); err != nil {
		return nil, fmt.Errorf("extractor emitted invalid JSON despite grammar: %w", err)
	}

	// pass 2: kind + source classification, one dedicated thinking call per fact
	for i := range facts {
		if k, src, force, err := e.classifyKindSource(facts[i].Statement, current); err == nil {
			if k != "" {
				facts[i].Kind = k
			}
			if src != "" {
				facts[i].Source = src
			}
			facts[i].Force = force
		} // on error: keep pass-1 values; classification is best-effort
	}
	return facts, nil
}

func (e *Extractor) Close() {
	if e.kind != e.extract {
		e.kind.Free()
	}
	e.extract.Free()
}

// ---- prompts and grammars (frozen from spike 3; hash-fingerprinted by bench) ----

const factGrammar = `
root      ::= "[" ws (fact (ws "," ws fact)*)? ws "]"
fact      ::= "{" ws "\"statement\"" ws ":" ws string ws "," ws "\"kind\"" ws ":" ws kind ws "," ws "\"source\"" ws ":" ws source ws "," ws "\"entities\"" ws ":" ws entities ws "," ws "\"confidence\"" ws ":" ws number ws "}"
kind      ::= "\"OBSERVED\"" | "\"DERIVED\"" | "\"PREFERENCE\"" | "\"CONSTRAINT\""
source    ::= "\"user\"" | "\"assistant\""
entities  ::= "[" ws (string (ws "," ws string)*)? ws "]"
string    ::= "\"" ([^"\\] | "\\" .)* "\""
number    ::= "0" ("." [0-9]+)? | "1" ("." "0"+)?
ws        ::= [ \t\n]*
`

// extraction system prompt v3 — best on the golden set at 1.7B (P=79.3%).
const extractSystemPrompt = `You extract durable facts from conversation turns for a working-memory store. Output ONLY a JSON array of fact objects: {"statement","kind","source","entities","confidence"}.

SOURCE — who ASSERTED the fact:
- "user": the fact's substance was stated by the user (decisions, corrections, orders, reports). If the assistant merely restates or confirms what the user said, the source is still "user".
- "assistant": the fact's substance was introduced by the assistant (its own observations, conclusions, plans).

WHAT COUNTS AS ONE FACT:
- One real-world fact = ONE entry: merge clauses about the SAME thing into one complete statement; decisions about DIFFERENT things are separate entries. Never split one fact, never fuse two.
- When the assistant merely restates, confirms, or acknowledges what the user said, that is the SAME fact — extract it once, not twice.
- General knowledge explanations (how something works in general) are NOT durable session facts. A turn that is question + textbook answer => [].
- QUESTIONS assert nothing: extract NO facts from a question — including facts the question presupposes ("why does the broker on li1 drop connections?" does NOT establish that the broker is on li1).
- For a REQUEST or ORDER ("make X do Y", "please change Z"), the fact is the operator's INTENT — phrase it "The operator wants …" — never as accomplished state.
- Transient chit-chat, greetings, scheduling small-talk => [].

KIND rubric — the DEFAULT is OBSERVED:
- OBSERVED: state, events, decisions, corrections, descriptions. When unsure, use OBSERVED.
- CONSTRAINT: ONLY a standing rule about how things MUST or MUST NEVER be done, phrased as law ("must never allocate", "no structure steeper than 30 degrees", "nothing below q8"). A decision or state is NOT a constraint.
- PREFERENCE: ONLY how the operator personally likes things done ("I prefer", "from now on do X", style/format/habit).
- DERIVED: ONLY a new conclusion or diagnosis reasoned out in this turn ("so the cause is X", "therefore I'll do Y"), not something directly stated.

Rules:
- statement: one short self-contained declarative sentence; resolve all pronouns using context.
- entities: kebab-case slugs. If a KNOWN ENTITIES slug applies, use it VERBATIM. Never invent spaced or capitalized names.
- Do not re-extract facts from PREVIOUS TURNS; they are context for pronoun resolution only.`

// kind+source classifier prompt v3 — decision tree; run with thinking ENABLED
// and no grammar (a grammar from token 1 suppresses Qwen3 thinking — spike 3).
// v3 adds SOURCE judgment: the 1.7B extractor attributes source by whose WORDS
// the statement echoes (44% on golden); assertion-provenance is a judgment call
// that belongs in this 4B pass.
const kindSystemPrompt = `Classify ONE extracted fact. Think briefly, then answer with exactly three words: KIND SOURCE FORCE.

Decide with two questions:

Q1 — Is it about the FUTURE (a standing rule for how things should be done from now on)?
  YES, and it is the operator's wish about workflow, style, format, or habits (how THEY like work done: "squash before pushing", "reports as tables", "run jobs overnight") => PREFERENCE
  YES, and it is a technical invariant of the SYSTEM (what the code/system must or must never do: "never allocate mid-tick", "max 30 degree slopes", "minimum q8") => CONSTRAINT
  NO => Q2

Q2 — Was it REASONED OUT in this turn, or directly stated?
  Someone concluded/diagnosed/planned it in this turn ("so the cause is...", "that means...", "therefore I'll...") => DERIVED
  Directly stated as fact, event, state, or past decision => OBSERVED

The operator saying what THEY want done = PREFERENCE even if phrased as "always/never". A rule the SYSTEM must obey = CONSTRAINT regardless of who said it.

Then decide SOURCE — who INTRODUCED the fact's substance, regardless of whose words the statement echoes:
  The user gave the decision/order/correction/report (even if the assistant restated or confirmed it) => user
  The assistant introduced it (its own observation, diagnosis, plan) => assistant

Then decide FORCE — what the speaker was DOING with the utterance the fact came from:
  DECISION: declaring, deciding, naming, or ruling — saying it MAKES it so ("we're calling it X", "the cap is 40, that's the rule")
  DIRECTIVE: asking for work or change ("make it...", "please add...") — the fact is the intent, not yet world state
  REPORT: describing existing world state ("the build is green", "I renamed it yesterday") — could be mistaken
  QUESTION: asking — nothing is asserted

Final answer format: KIND SOURCE FORCE (e.g. "CONSTRAINT user DECISION" or "OBSERVED assistant REPORT").`

func buildExtractionPrompt(current Turn, context []Turn, glossary map[string]string) string {
	var b strings.Builder
	b.WriteString("<|im_start|>system\n" + extractSystemPrompt + "<|im_end|>\n<|im_start|>user\n")
	if len(glossary) > 0 {
		b.WriteString("KNOWN ENTITIES:\n")
		keys := make([]string, 0, len(glossary))
		for k := range glossary {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(&b, "- %s: %s\n", k, glossary[k])
		}
		b.WriteString("\n")
	}
	if len(context) > 0 {
		b.WriteString("PREVIOUS TURNS (context only, do not re-extract):\n")
		for _, t := range context {
			fmt.Fprintf(&b, "[user]: %s\n[assistant]: %s\n", t.User, t.Assistant)
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "CURRENT TURN (extract from this):\n[user]: %s\n[assistant]: %s<|im_end|>\n", current.User, current.Assistant)
	b.WriteString("<|im_start|>assistant\n<think>\n\n</think>\n\n")
	return b.String()
}

const pairSystemPrompt = `Two statements about the same topic. Think briefly, then answer with exactly one word:
- SAME: they assert the same fact (different wording is fine)
- CONTRADICTS: they cannot both be true (a value, place, or polarity differs) — OR one is a standing rule/constraint and the other asks or intends to violate it
- DISTINCT: compatible but different facts about the topic`

// JudgePair classifies the relation between a new statement and an existing
// fact. Runs on the kind/judgment model (4B, thinking enabled) — calibration
// showed embeddings cannot make this call (dup and contradiction cosines
// overlap almost completely; a contradiction IS a near-paraphrase).
func (e *Extractor) JudgePair(a, b string) (PairVerdict, error) {
	prompt := "<|im_start|>system\n" + pairSystemPrompt + "<|im_end|>\n<|im_start|>user\nA: " + a + "\nB: " + b + "\n\nVerdict?<|im_end|>\n<|im_start|>assistant\n"
	ctx, err := e.kind.NewContext(2048, e.threads)
	if err != nil {
		return "", err
	}
	defer ctx.Free()
	out, _, err := ctx.Generate(prompt, "", 512)
	if err != nil {
		return "", err
	}
	verdict := out
	if i := strings.LastIndex(out, "</think>"); i >= 0 {
		verdict = out[i+len("</think>"):]
	}
	for _, v := range []PairVerdict{PairContradicts, PairDistinct, PairSame} {
		if strings.Contains(verdict, string(v)) {
			return v, nil
		}
	}
	return "", fmt.Errorf("no pair verdict")
}

func (e *Extractor) classifyKindSource(statement string, turn Turn) (string, string, string, error) {
	prompt := "<|im_start|>system\n" + kindSystemPrompt + "<|im_end|>\n<|im_start|>user\n" +
		"TURN IT CAME FROM:\n[user]: " + turn.User + "\n[assistant]: " + turn.Assistant +
		"\n\nFACT: " + statement + "\n\nKind?<|im_end|>\n<|im_start|>assistant\n"
	ctx, err := e.kind.NewContext(2048, e.threads)
	if err != nil {
		return "", "", "", err
	}
	defer ctx.Free()
	out, _, err := ctx.Generate(prompt, "", 512)
	if err != nil {
		return "", "", "", err
	}
	verdict := out
	if i := strings.LastIndex(out, "</think>"); i >= 0 {
		verdict = out[i+len("</think>"):]
	}
	kind, source, force := "", "", ""
	for _, k := range []string{"CONSTRAINT", "PREFERENCE", "DERIVED", "OBSERVED"} {
		if strings.Contains(verdict, k) {
			kind = k
			break
		}
	}
	lower := strings.ToLower(verdict)
	if strings.Contains(lower, "user") {
		source = "user"
	} else if strings.Contains(lower, "assistant") {
		source = "assistant"
	}
	for _, f := range []string{ForceQuestion, ForceDirective, ForceDecision, ForceReport} {
		if strings.Contains(verdict, f) {
			force = f
			break
		}
	}
	if kind == "" && source == "" {
		return "", "", "", fmt.Errorf("no verdict")
	}
	return kind, source, force, nil
}
