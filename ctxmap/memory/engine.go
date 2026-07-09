// Package memory is the ctxmap engine: the harness-agnostic core that a host
// harness (bridle, the agora research harness, anything with a turn loop)
// drives. The host owns the model call and the turn loop; the engine owns
// everything memory: prompt-block assembly, retrieval, the recall/inspect
// tools, async extraction, reconciliation, and the store lifecycle.
//
// Host contract per turn:
//
//	blocks := eng.AssembleBlocks(userMsg, turnN)   // before the model call
//	... host builds prompt: [system + blocks.Framing + blocks.Core]
//	    [tail] [blocks.Subgraph] [userMsg], serves eng.Tools() ...
//	eng.RecordTurn(turnN, userMsg, answer, blocks.RenderedIDs) // after
//
// Extraction never blocks a turn: it runs on a background worker, and a dead
// or slow extractor degrades the host to plain-transcript behavior.
package memory

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/CarriedWorldUniverse/bridle/ctxmap/embed"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/extractor"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/render"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/store"
)

// Proposer extracts fact proposals from a turn. *extractor.Extractor (behind
// the ctxmap_llama build tag) satisfies it.
type Proposer interface {
	Propose(current extractor.Turn, ctx []extractor.Turn, glossary map[string]string) ([]extractor.FactProposal, error)
}

// PairJudge classifies two same-topic statements (SAME/CONTRADICTS/DISTINCT).
type PairJudge interface {
	JudgePair(a, b string) (extractor.PairVerdict, error)
}

type Config struct {
	SessionID    string // scopes PROPOSED-fact visibility (store session rule)
	ContextTurns int    // turns of context handed to the extractor (default 2)
}

// Tool is a neutral tool descriptor the host maps into its own tool schema.
type Tool struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Run         func(args json.RawMessage) string
}

// Blocks is what AssembleBlocks returns for prompt construction.
type Blocks struct {
	Framing     string   // memory-is-automatic framing (stable)
	Core        string   // epoch-frozen verified core (stable within epoch)
	Subgraph    string   // per-turn relevant facts — place at the prompt END
	RenderedIDs []string // pass to RecordTurn for reuse-confirmation credit
	Notices     []string // contradiction notices surfaced this turn
}

type turnRec struct{ user, assistant string }

type Engine struct {
	mu    sync.Mutex
	cfg   Config
	st    *store.Store
	rend  *render.Renderer
	prop  Proposer
	judge PairJudge
	emb   embed.Embedder

	turns    []turnRec // engine-owned dialogue record (inspect evidence, extractor context)
	extractQ chan extractReq
	wg       sync.WaitGroup
	lastIDs  []string
	pending  int
}

type extractReq struct {
	turnN       int
	renderedIDs []string
}

// New creates an engine. prop may be nil (extraction disabled — read-only
// memory over an existing store). emb/judge may be nil (reconciliation falls
// back to token heuristics).
func New(cfg Config, st *store.Store, rend *render.Renderer, prop Proposer, emb embed.Embedder, judge PairJudge) *Engine {
	if cfg.ContextTurns == 0 {
		cfg.ContextTurns = 2
	}
	if cfg.SessionID == "" {
		cfg.SessionID = fmt.Sprintf("ctx_%d", time.Now().UnixMilli())
	}
	e := &Engine{cfg: cfg, st: st, rend: rend, prop: prop, emb: emb, judge: judge,
		extractQ: make(chan extractReq, 64)}
	e.wg.Add(1)
	go e.worker()
	return e
}

func (e *Engine) SessionID() string { return e.cfg.SessionID }

// Close drains the extraction queue and stops the worker.
func (e *Engine) Close() {
	close(e.extractQ)
	e.wg.Wait()
}

// Framing tells the model memory is automatic — omit it and models fixate on
// being unable to "save" facts (measured; see the write-up).
const Framing = `## Working memory (automatic)
Durable facts from this conversation are captured for you automatically in the background — you never save, persist, or write anything yourself, and you have no tool to do so. Facts already known appear under "Working memory" below; treat them as established context. Just converse naturally: when the user tells you something, respond to its substance — do not acknowledge it as "saved" or apologize for being unable to save it. Use the ` + "`recall`" + ` tool ONLY when you need older context that is not visible in the prompt, and ` + "`inspect`" + ` only to check a fact's evidence. Never call them just to verify that something was stored.
BEFORE acting on a request, check it against the constraints shown in working memory. If the request conflicts with a constraint — or a NOTICE line reports a conflict — raise it to the user ONCE, plainly, citing both sides, and do NOT do the conflicting work until they answer. Then commit to their answer fully, without relitigating.`

// AssembleBlocks prepares the memory blocks for the next turn. It also
// auto-consolidates (re-renders the core) when the verified set moved —
// cache stability is a within-turn property; turn boundaries reseed cheaply.
func (e *Engine) AssembleBlocks(userMsg string, turnN int) Blocks {
	e.mu.Lock()
	defer e.mu.Unlock()
	if stale, err := e.rend.CoreStale(); err == nil && stale {
		e.rend.NewEpoch()
	}
	b := Blocks{Framing: Framing, Core: e.rend.RenderCore()}
	seeds := e.retrieve(userMsg)
	sub, ids := e.rend.RenderSubgraph(seeds)
	if strings.TrimSpace(sub) != "" {
		b.Subgraph = sub
	}
	b.RenderedIDs = ids
	for _, line := range strings.Split(sub, "\n") {
		if strings.Contains(line, "NOTICE:") {
			b.Notices = append(b.Notices, strings.TrimPrefix(strings.TrimSpace(line), "- "))
		}
	}
	// synthetic in-turn conflict notice: extraction-produced CONTRADICTS links
	// land one turn late (async), and an ambient constraint line in the
	// subgraph measurably loses to task momentum (challenge bench: 0-1/3).
	// When a VERIFIED CONSTRAINT overlaps the incoming message, say so
	// EXPLICITLY, adjacent to the request.
	for _, f := range seeds {
		if f.Kind == store.KindConstraint && f.Status == store.StatusVerified && overlapF1(f.Statement, userMsg) >= 0.25 {
			n := fmt.Sprintf("NOTICE: this request may conflict with the established constraint [%s] %q — check before acting, and raise the conflict with the user if it holds.", f.ID, f.Statement)
			b.Notices = append(b.Notices, n)
			b.Subgraph = strings.TrimRight(b.Subgraph, "\n") + "\n- " + n + "\n"
		}
	}
	return b
}

// RecordTurn feeds the completed turn back: reuse-confirmation credit for
// rendered facts, the engine's dialogue record, and async extraction.
func (e *Engine) RecordTurn(turnN int, user, assistant string, renderedIDs []string) {
	e.mu.Lock()
	for _, id := range renderedIDs {
		e.st.RecordRender(id, turnN)
	}
	for len(e.turns) < turnN {
		e.turns = append(e.turns, turnRec{})
	}
	e.turns[turnN-1] = turnRec{user: user, assistant: assistant}
	e.mu.Unlock()

	if e.prop == nil {
		return
	}
	select {
	case e.extractQ <- extractReq{turnN: turnN, renderedIDs: renderedIDs}:
		e.mu.Lock()
		e.pending++
		e.mu.Unlock()
	default: // queue full: degrade to plain transcript rather than block
	}
}

// WaitExtraction blocks until queued extractions land; returns the ids
// asserted by the most recent completed extraction.
func (e *Engine) WaitExtraction() []string {
	for {
		e.mu.Lock()
		done := e.pending == 0
		ids := append([]string{}, e.lastIDs...)
		e.mu.Unlock()
		if done {
			return ids
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// Tools returns the recall/inspect tools for the host to serve to the model.
func (e *Engine) Tools() []Tool {
	return []Tool{
		{
			Name:        "recall",
			Description: "Retrieve OLDER facts from earlier in this conversation that are not shown in the current prompt. Only needed when answering requires context beyond what is visible. Facts are stored automatically; this only reads them.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","description":"what to search for"}},"required":["query"]}`),
			Run:         e.runRecall,
		},
		{
			Name:        "inspect",
			Description: "Show the original transcript evidence behind one fact id (for auditing why a fact is believed). Rarely needed.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"fact_id":{"type":"string"}},"required":["fact_id"]}`),
			Run:         e.runInspect,
		},
	}
}

func (e *Engine) runRecall(args json.RawMessage) string {
	var a struct {
		Query string `json:"query"`
	}
	json.Unmarshal(args, &a)
	e.mu.Lock()
	defer e.mu.Unlock()
	seeds := e.retrieve(a.Query)
	text, ids := e.rend.RenderRecall(seeds)
	turn := len(e.turns) + 1
	for _, id := range ids {
		e.st.RecordRender(id, turn)
	}
	return text
}

func (e *Engine) runInspect(args json.RawMessage) string {
	var a struct {
		FactID string `json:"fact_id"`
	}
	json.Unmarshal(args, &a)
	f, err := e.st.Get(a.FactID)
	if err != nil {
		return "no such fact: " + a.FactID
	}
	ev := "(transcript evidence unavailable)"
	e.mu.Lock()
	for _, sp := range f.Provenance {
		if sp.Turn-1 >= 0 && sp.Turn-1 < len(e.turns) {
			t := e.turns[sp.Turn-1]
			ev = fmt.Sprintf("turn %d — [user]: %s\n[assistant]: %s", sp.Turn, t.user, t.assistant)
			break
		}
	}
	e.mu.Unlock()
	return fmt.Sprintf("[%s] %s (kind=%s status=%s trust=%s)\nEVIDENCE:\n%s", f.ID, f.Statement, f.Kind, f.Status, f.Trust, ev)
}

// Preview renders what the next turn's subgraph would contain for a
// hypothetical message — no model call, no RecordRender bookkeeping.
// Debug/eval surface (ctxmapd render_preview).
func (e *Engine) Preview(msg string) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	sub, _ := e.rend.RenderSubgraph(e.retrieve(msg))
	return e.rend.RenderCore() + "\n\n" + sub
}

// ---- retrieval (word-match fallback; embedding retrieval is a future unit) ----

var wordRe = regexp.MustCompile(`[a-zA-Z0-9][a-zA-Z0-9_-]{3,}`)

func (e *Engine) retrieve(msg string) []*store.Fact {
	seen := map[string]bool{}
	var seeds []*store.Fact
	add := func(fs []*store.Fact) {
		for _, f := range fs {
			if !seen[f.ID] {
				seen[f.ID] = true
				seeds = append(seeds, f)
			}
		}
	}
	text := msg
	for i := len(e.turns) - 1; i >= 0 && i >= len(e.turns)-e.cfg.ContextTurns; i-- {
		text += " " + e.turns[i].user
	}
	words := wordRe.FindAllString(text, 12)
	// constraints first: a VERIFIED CONSTRAINT sharing words with the incoming
	// message must be in front of the model AT THE VIOLATING TURN — conflict
	// links from extraction land one turn late by construction (async), so
	// in-turn challenge depends on retrieval, not on notices (measured:
	// challenge rate 1/3 without this priority)
	for _, w := range words {
		if fs, err := e.st.QueryText(w, 3, e.cfg.SessionID); err == nil {
			for _, f := range fs {
				if f.Kind == store.KindConstraint && f.Status == store.StatusVerified {
					add([]*store.Fact{f})
				}
			}
		}
	}
	for _, w := range words {
		if fs, err := e.st.QueryText(w, 3, e.cfg.SessionID); err == nil {
			add(fs)
		}
		if fs, err := e.st.QueryEntity(strings.ToLower(w), 3, e.cfg.SessionID); err == nil {
			add(fs)
		}
	}
	if len(seeds) > 12 {
		seeds = seeds[:12]
	}
	return seeds
}

// ---- extraction worker + reconciliation (ported from the research harness;
// see the write-up for the calibration behind the thresholds) ----

const (
	sameTopicCos = 0.80
	tokenDupF1   = 0.90
)

func (e *Engine) worker() {
	defer e.wg.Done()
	for req := range e.extractQ {
		e.extractTurn(req.turnN)
		e.mu.Lock()
		e.pending--
		e.mu.Unlock()
	}
}

func (e *Engine) extractTurn(turnN int) {
	e.mu.Lock()
	if turnN-1 >= len(e.turns) {
		e.mu.Unlock()
		return
	}
	cur := e.turns[turnN-1]
	curUser, curAsst := capText(cur.user, 4000), capText(cur.assistant, 4000)
	var ctxTurns []extractor.Turn
	for i := turnN - 1 - e.cfg.ContextTurns; i < turnN-1; i++ {
		if i >= 0 {
			ctxTurns = append(ctxTurns, extractor.Turn{User: capText(e.turns[i].user, 1200), Assistant: capText(e.turns[i].assistant, 1200)})
		}
	}
	glossary := e.glossary()
	e.mu.Unlock()

	props, err := e.prop.Propose(extractor.Turn{User: curUser, Assistant: curAsst}, ctxTurns, glossary)
	if err != nil {
		return // extractor failure degrades to plain transcript
	}

	var ids []string
	for _, p := range props {
		// questions assert nothing — a fact judged to come from a question is
		// a presupposition leak, the poisoning vector the force pass guards
		if p.Force == extractor.ForceQuestion {
			continue
		}
		kind := store.Kind(p.Kind)
		trust := store.TrustModelObserved
		if kind == store.KindDerived {
			trust = store.TrustModelDerived
		}
		// model-proposed source, deterministically ground-checked: model
		// say-so never mints operator trust
		performative := false
		// directives are phrased as intent ("The operator wants …") — strip
		// the wrapper before grounding, since those words are never in the
		// user's own text
		groundStmt := p.Statement
		if p.Force == extractor.ForceDirective {
			low := strings.ToLower(groundStmt)
			for _, pre := range []string{"the operator wants ", "operator wants "} {
				if strings.HasPrefix(low, pre) {
					groundStmt = groundStmt[len(pre):]
					break
				}
			}
		}
		if p.Source == "user" && kind != store.KindDerived && groundedInText(groundStmt, curUser) {
			trust = store.TrustOperatorStated
			// decisions and directives are performative (saying makes it so:
			// the rule exists / the intent is real). REPORTs describe world
			// state the operator can be wrong about: top trust rank for
			// conflicts, but PROPOSED entry — promotion via reuse or pin.
			// Unknown force degrades conservatively to REPORT semantics.
			performative = p.Force == extractor.ForceDecision || p.Force == extractor.ForceDirective
		}
		f := store.Fact{
			Statement: p.Statement, Kind: kind, Trust: trust, Confidence: p.Confidence, Performative: performative,
			Entities: p.Entities, SessionID: e.cfg.SessionID,
			Provenance: []store.Span{{SessionID: e.cfg.SessionID, Turn: turnN, Start: 0, End: len(cur.user) + len(cur.assistant)}},
		}
		if dupID, contraID := e.reconcileScan(p.Statement, p.Entities); dupID != "" {
			e.st.RecordRender(dupID, turnN) // re-observation confirms
			continue
		} else if contraID != "" {
			if id, err := e.st.AssertFact(f); err == nil {
				if p.Force == extractor.ForceDirective {
					// an ASK that conflicts with an established fact/rule is
					// surfaced, never silently resolved: the newer-operator-
					// statement-wins rule would retract the constraint the
					// operator themselves verified. Link -> notice -> the
					// model raises it (challenge principle).
					e.st.Link(id, contraID, store.LinkContradicts)
				} else {
					e.st.ResolveContradiction(id, contraID)
				}
				ids = append(ids, id)
				e.saveEmbedding(id, p.Statement)
			}
			continue
		}
		if kind == store.KindDerived {
			if pid := e.recentEntityFact(p.Entities); pid != "" {
				f.Parents = []string{pid}
			} else {
				f.Kind, f.Trust = store.KindObserved, store.TrustModelObserved
			}
		}
		if id, err := e.st.AssertFact(f); err == nil {
			ids = append(ids, id)
			e.saveEmbedding(id, p.Statement)
		}
	}
	e.mu.Lock()
	e.lastIDs = ids
	e.mu.Unlock()
}

func (e *Engine) reconcileScan(statement string, entities []string) (string, string) {
	if e.emb == nil || e.judge == nil {
		return e.reconcileScanTokens(statement, entities)
	}
	vec, err := e.emb.Embed(statement)
	if err != nil {
		return e.reconcileScanTokens(statement, entities)
	}
	all, err := e.st.Embeddings(e.cfg.SessionID)
	if err != nil {
		return e.reconcileScanTokens(statement, entities)
	}
	type cand struct {
		id  string
		cos float64
	}
	var cands []cand
	for id, v := range all {
		if c := embed.Cos(vec, v); c >= sameTopicCos {
			cands = append(cands, cand{id, c})
		}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].cos > cands[j].cos })
	if len(cands) > 3 {
		cands = cands[:3]
	}
	for _, c := range cands {
		f, err := e.st.Get(c.id)
		if err != nil || f.Status == store.StatusRetracted {
			continue
		}
		if overlapF1(statement, f.Statement) >= tokenDupF1 {
			return f.ID, ""
		}
		verdict, err := e.judge.JudgePair(statement, f.Statement)
		if err != nil {
			continue
		}
		switch verdict {
		case extractor.PairSame:
			return f.ID, ""
		case extractor.PairContradicts:
			return "", f.ID
		}
	}
	return "", ""
}

func (e *Engine) reconcileScanTokens(statement string, entities []string) (string, string) {
	seen := map[string]bool{}
	var cands []*store.Fact
	for _, ent := range entities {
		if fs, err := e.st.QueryEntity(ent, 8, e.cfg.SessionID); err == nil {
			for _, f := range fs {
				if !seen[f.ID] {
					seen[f.ID] = true
					cands = append(cands, f)
				}
			}
		}
	}
	for _, f := range cands {
		ov := overlapF1(statement, f.Statement)
		if ov >= 0.9 {
			return f.ID, ""
		}
		if ov >= 0.55 {
			return "", f.ID
		}
	}
	return "", ""
}

func (e *Engine) recentEntityFact(entities []string) string {
	for _, ent := range entities {
		if fs, err := e.st.QueryEntity(ent, 1, e.cfg.SessionID); err == nil && len(fs) > 0 {
			return fs[0].ID
		}
	}
	return ""
}

func (e *Engine) glossary() map[string]string {
	out := map[string]string{}
	core, err := e.st.Core()
	if err != nil {
		return out
	}
	for _, f := range core {
		for _, ent := range f.Entities {
			if _, ok := out[ent]; !ok {
				words := strings.Fields(f.Statement)
				if len(words) > 10 {
					words = words[:10]
				}
				out[ent] = strings.Join(words, " ")
			}
		}
	}
	return out
}

func (e *Engine) saveEmbedding(id, statement string) {
	if e.emb == nil {
		return
	}
	if vec, err := e.emb.Embed(statement); err == nil {
		e.st.SetEmbedding(id, vec)
	}
}

// ---- text helpers ----

func capText(s string, max int) string {
	if len(s) <= max {
		return s
	}
	half := max / 2
	return s[:half] + "\n[…truncated…]\n" + s[len(s)-half:]
}

func groundedInText(statement, text string) bool {
	stmt, src := tokset(statement), tokset(text)
	if len(stmt) == 0 {
		return false
	}
	hit := 0
	for w := range stmt {
		if src[w] || src[stem(w)] || srcHasStem(src, w) {
			hit++
		}
	}
	return float64(hit)/float64(len(stmt)) >= 0.6
}

// stem: crude suffix strip so "raised"/"raise", "caps"/"cap" ground each other.
func stem(w string) string {
	for _, suf := range []string{"ed", "es", "s", "ing"} {
		if strings.HasSuffix(w, suf) && len(w)-len(suf) >= 4 {
			return w[:len(w)-len(suf)]
		}
	}
	return w
}

func srcHasStem(src map[string]bool, w string) bool {
	ws := stem(w)
	for s := range src {
		ss := stem(s)
		// prefix-tolerant: "rais"(<-raised) matches "raise"; both directions
		if ss == ws || (len(ws) >= 4 && strings.HasPrefix(ss, ws)) || (len(ss) >= 4 && strings.HasPrefix(ws, ss)) {
			return true
		}
	}
	return false
}

func tokset(s string) map[string]bool {
	out := map[string]bool{}
	for _, w := range wordRe.FindAllString(strings.ToLower(s), -1) {
		out[w] = true
	}
	return out
}

func overlapF1(a, b string) float64 {
	A, B := tokset(a), tokset(b)
	if len(A) == 0 || len(B) == 0 {
		return 0
	}
	inter := 0
	for w := range A {
		if B[w] {
			inter++
		}
	}
	if inter == 0 {
		return 0
	}
	p, r := float64(inter)/float64(len(B)), float64(inter)/float64(len(A))
	return 2 * p * r / (p + r)
}
