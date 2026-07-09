// Package store is the ctxmap fact store: SQLite-backed, provenance-mandatory.
//
// P2 rules enforced at this boundary (spec §4):
//   - a Fact cannot be asserted without ≥1 provenance span
//   - a DERIVED fact cannot be asserted without parents
//   - only the reconciler paths in this package change status; callers
//     (extractor, renderer) get no direct status writes
//
// Lifecycle (spec §3): PROPOSED → VERIFIED via operator pin, operator-stated
// entry, or reuse-confirmation (rendered into ≥K distinct turns without a
// CONTRADICTS link). Contradictions resolve by trust rank; operator wins.
package store

import (
	"crypto/rand"
	"database/sql"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type Kind string
type Status string
type Trust string
type LinkType string

const (
	KindObserved   Kind = "OBSERVED"
	KindDerived    Kind = "DERIVED"
	KindPreference Kind = "PREFERENCE"
	KindConstraint Kind = "CONSTRAINT"

	StatusProposed  Status = "PROPOSED"
	StatusVerified  Status = "VERIFIED"
	StatusRetracted Status = "RETRACTED"

	TrustOperatorStated Trust = "OPERATOR_STATED"
	TrustModelObserved  Trust = "MODEL_OBSERVED"
	TrustModelDerived   Trust = "MODEL_DERIVED"

	LinkSupports    LinkType = "SUPPORTS"
	LinkContradicts LinkType = "CONTRADICTS"
	LinkRefines     LinkType = "REFINES"
	LinkDerivedFrom LinkType = "DERIVED_FROM"
	LinkSameEntity  LinkType = "ABOUT_SAME_ENTITY"
)

// ReuseConfirmK: rendered into context in >=K distinct turns without acquiring
// a CONTRADICTS link => auto-promote to VERIFIED (spec §7, K=3 v0).
const ReuseConfirmK = 3

func trustRank(t Trust) int {
	switch t {
	case TrustOperatorStated:
		return 3
	case TrustModelObserved:
		return 2
	case TrustModelDerived:
		return 1
	}
	return 0
}

// Span points into the transcript: the ground truth this fact interprets.
type Span struct {
	SessionID string `json:"session_id"`
	Turn      int    `json:"turn"`
	Start     int    `json:"start"`
	End       int    `json:"end"`
}

type Fact struct {
	ID         string
	Statement  string
	Kind       Kind
	Status     Status
	Stale      bool
	Pinned     bool
	Trust      Trust
	Confidence float64
	Entities   []string
	Parents    []string
	Provenance []Span
	// Performative marks facts whose utterance MAKES them true (decisions,
	// rules, namings, stated intents). Only performative operator-stated
	// facts enter VERIFIED on assertion; operator world-state REPORTS keep
	// top trust rank for conflicts but enter PROPOSED — VERIFIED means
	// "safe to build on", not "the operator said it".
	Performative  bool
	SessionID     string
	Created       time.Time
	LastConfirmed time.Time
	RenderTurns   []int // distinct turns this fact was rendered into context
}

type Store struct {
	db *sql.DB
}

const schema = `
CREATE TABLE IF NOT EXISTS facts (
	id TEXT PRIMARY KEY,
	statement TEXT NOT NULL,
	kind TEXT NOT NULL,
	status TEXT NOT NULL,
	stale INTEGER NOT NULL DEFAULT 0,
	pinned INTEGER NOT NULL DEFAULT 0,
	trust TEXT NOT NULL,
	confidence REAL NOT NULL,
	entities TEXT NOT NULL,       -- JSON array of slugs
	parents TEXT NOT NULL,        -- JSON array of fact ids
	embedding BLOB,               -- reserved for the embedder (not wired in v0 step 1)
	session_id TEXT NOT NULL,
	created TEXT NOT NULL,
	last_confirmed TEXT NOT NULL,
	render_turns TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS links (
	from_id TEXT NOT NULL,
	to_id TEXT NOT NULL,
	type TEXT NOT NULL,
	created TEXT NOT NULL,
	PRIMARY KEY (from_id, to_id, type)
);
CREATE TABLE IF NOT EXISTS provenance (
	fact_id TEXT NOT NULL,
	session_id TEXT NOT NULL,
	turn INTEGER NOT NULL,
	start INTEGER NOT NULL,
	end INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status, stale);
CREATE INDEX IF NOT EXISTS idx_prov_fact ON provenance(fact_id);
`

func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, err
	}
	// single connection: SQLite is single-writer, and :memory: DBs are
	// per-connection — pooling would split the database.
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, err
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() error { return s.db.Close() }

func newID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return fmt.Sprintf("f_%x_%s", time.Now().UnixMilli(), hex.EncodeToString(b)[:6])
}

// secretPattern: deny-pass at the store boundary (spec §7). Credential-shaped
// content must never enter the map.
var secretPatterns = []*regexp.Regexp{
	regexp.MustCompile(`ghp_[A-Za-z0-9]{20,}`),
	regexp.MustCompile(`github_pat_[A-Za-z0-9_]{20,}`),
	regexp.MustCompile(`sk-[A-Za-z0-9_-]{20,}`),
	regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----`),
	regexp.MustCompile(`AKIA[0-9A-Z]{16}`),
	regexp.MustCompile(`eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}`), // JWT
	regexp.MustCompile(`\b[A-Za-z0-9+/=]{40,}\b`),                      // long high-entropy-ish blob
}

func containsSecret(text string) bool {
	for _, p := range secretPatterns {
		if p.MatchString(text) {
			return true
		}
	}
	return false
}

// AssertFact stores a new fact. Enforces the P2 construction rules and the
// secret deny-pass. Operator-stated facts enter at VERIFIED; all else PROPOSED.
func (s *Store) AssertFact(f Fact) (string, error) {
	if strings.TrimSpace(f.Statement) == "" {
		return "", fmt.Errorf("empty statement")
	}
	if len(f.Provenance) == 0 {
		return "", fmt.Errorf("fact rejected: no provenance span (no just-trust-me facts)")
	}
	if f.Kind == KindDerived && len(f.Parents) == 0 {
		return "", fmt.Errorf("fact rejected: DERIVED without parents")
	}
	if containsSecret(f.Statement) {
		return "", fmt.Errorf("fact rejected: credential-shaped content")
	}
	if trustRank(f.Trust) == 0 {
		return "", fmt.Errorf("fact rejected: unknown trust %q", f.Trust)
	}
	f.ID = newID()
	f.Status = StatusProposed
	if f.Trust == TrustOperatorStated && f.Performative {
		f.Status = StatusVerified
	}
	now := time.Now().UTC()
	f.Created, f.LastConfirmed = now, now

	ents, _ := json.Marshal(f.Entities)
	pars, _ := json.Marshal(f.Parents)
	rts, _ := json.Marshal([]int{})
	tx, err := s.db.Begin()
	if err != nil {
		return "", err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`INSERT INTO facts (id,statement,kind,status,stale,pinned,trust,confidence,entities,parents,session_id,created,last_confirmed,render_turns)
		VALUES (?,?,?,?,0,?,?,?,?,?,?,?,?,?)`,
		f.ID, f.Statement, string(f.Kind), string(f.Status), boolInt(f.Pinned), string(f.Trust), f.Confidence,
		string(ents), string(pars), f.SessionID, now.Format(time.RFC3339), now.Format(time.RFC3339), string(rts)); err != nil {
		return "", err
	}
	for _, sp := range f.Provenance {
		if _, err := tx.Exec(`INSERT INTO provenance (fact_id,session_id,turn,start,end) VALUES (?,?,?,?,?)`,
			f.ID, sp.SessionID, sp.Turn, sp.Start, sp.End); err != nil {
			return "", err
		}
	}
	for _, p := range f.Parents {
		if _, err := tx.Exec(`INSERT OR IGNORE INTO links (from_id,to_id,type,created) VALUES (?,?,?,?)`,
			f.ID, p, string(LinkDerivedFrom), now.Format(time.RFC3339)); err != nil {
			return "", err
		}
	}
	return f.ID, tx.Commit()
}

func (s *Store) Link(from, to string, t LinkType) error {
	_, err := s.db.Exec(`INSERT OR IGNORE INTO links (from_id,to_id,type,created) VALUES (?,?,?,?)`,
		from, to, string(t), time.Now().UTC().Format(time.RFC3339))
	return err
}

// Pin marks a fact operator-pinned and promotes it.
func (s *Store) Pin(id string) error {
	res, err := s.db.Exec(`UPDATE facts SET pinned=1, status=? WHERE id=? AND status!=?`,
		string(StatusVerified), id, string(StatusRetracted))
	return execErr(res, err, id)
}

func (s *Store) Retract(id string) error {
	res, err := s.db.Exec(`UPDATE facts SET status=? WHERE id=?`, string(StatusRetracted), id)
	if err == nil {
		// a retracted parent flags all descendants stale (spec §8)
		s.db.Exec(`UPDATE facts SET stale=1 WHERE id IN (SELECT from_id FROM links WHERE to_id=? AND type=?)`,
			id, string(LinkDerivedFrom))
	}
	return execErr(res, err, id)
}

// ResolveContradiction applies the trust rule for newFact contradicting oldFact:
// pinned old fact => flag only; strictly-higher trust OR operator-vs-operator
// => auto-retract old; otherwise flag only. Returns true if old was retracted.
func (s *Store) ResolveContradiction(newID, oldID string) (bool, error) {
	nf, err := s.Get(newID)
	if err != nil {
		return false, err
	}
	of, err := s.Get(oldID)
	if err != nil {
		return false, err
	}
	if err := s.Link(newID, oldID, LinkContradicts); err != nil {
		return false, err
	}
	if of.Pinned {
		return false, nil // pinned: never auto-retract, notice surfaces via the CONTRADICTS link
	}
	nr, or := trustRank(nf.Trust), trustRank(of.Trust)
	if nr > or || (nf.Trust == TrustOperatorStated && of.Trust == TrustOperatorStated) {
		return true, s.Retract(oldID)
	}
	return false, nil
}

// RecordRender notes that a fact was rendered into context at turn n and
// applies the reuse-confirmation promotion rule.
func (s *Store) RecordRender(id string, turn int) error {
	f, err := s.Get(id)
	if err != nil {
		return err
	}
	for _, t := range f.RenderTurns {
		if t == turn {
			return nil
		}
	}
	f.RenderTurns = append(f.RenderTurns, turn)
	rts, _ := json.Marshal(f.RenderTurns)
	now := time.Now().UTC().Format(time.RFC3339)
	if _, err := s.db.Exec(`UPDATE facts SET render_turns=?, last_confirmed=? WHERE id=?`, string(rts), now, id); err != nil {
		return err
	}
	if f.Status == StatusProposed && len(f.RenderTurns) >= ReuseConfirmK {
		var contradicted int
		s.db.QueryRow(`SELECT COUNT(*) FROM links WHERE (from_id=? OR to_id=?) AND type=?`,
			id, id, string(LinkContradicts)).Scan(&contradicted)
		if contradicted == 0 {
			_, err = s.db.Exec(`UPDATE facts SET status=? WHERE id=?`, string(StatusVerified), id)
			return err
		}
	}
	return nil
}

func (s *Store) Get(id string) (*Fact, error) {
	row := s.db.QueryRow(`SELECT id,statement,kind,status,stale,pinned,trust,confidence,entities,parents,session_id,created,last_confirmed,render_turns FROM facts WHERE id=?`, id)
	f, err := scanFact(row)
	if err != nil {
		return nil, err
	}
	rows, err := s.db.Query(`SELECT session_id,turn,start,end FROM provenance WHERE fact_id=?`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var sp Span
		if err := rows.Scan(&sp.SessionID, &sp.Turn, &sp.Start, &sp.End); err != nil {
			return nil, err
		}
		f.Provenance = append(f.Provenance, sp)
	}
	return f, rows.Err()
}

// Session visibility (spec: per-project store, per-session working set):
// VERIFIED facts are project-wide; PROPOSED facts are visible only to the
// session that asserted them. sessionID "" = no scoping (admin/control view).
const sessionScope = ` AND (status=? OR session_id=? OR ?='')`

// QueryText: LIKE-based retrieval fallback until the embedder lands.
// sessionID scopes visibility; pass "" for the unscoped admin view.
func (s *Store) QueryText(q string, limit int, sessionID string) ([]*Fact, error) {
	rows, err := s.db.Query(`SELECT id,statement,kind,status,stale,pinned,trust,confidence,entities,parents,session_id,created,last_confirmed,render_turns
		FROM facts WHERE status!=? AND statement LIKE ?`+sessionScope+` ORDER BY last_confirmed DESC LIMIT ?`,
		string(StatusRetracted), "%"+q+"%", string(StatusVerified), sessionID, sessionID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanFacts(rows)
}

func (s *Store) QueryEntity(slug string, limit int, sessionID string) ([]*Fact, error) {
	rows, err := s.db.Query(`SELECT id,statement,kind,status,stale,pinned,trust,confidence,entities,parents,session_id,created,last_confirmed,render_turns
		FROM facts WHERE status!=? AND entities LIKE ?`+sessionScope+` ORDER BY last_confirmed DESC LIMIT ?`,
		string(StatusRetracted), `%"`+slug+`"%`, string(StatusVerified), sessionID, sessionID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanFacts(rows)
}

// Core returns the core-map population: VERIFIED, not stale, not retracted.
func (s *Store) Core() ([]*Fact, error) {
	rows, err := s.db.Query(`SELECT id,statement,kind,status,stale,pinned,trust,confidence,entities,parents,session_id,created,last_confirmed,render_turns
		FROM facts WHERE status=? AND stale=0 ORDER BY created`, string(StatusVerified))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanFacts(rows)
}

func (s *Store) Neighbors(id string, depth int) ([]*Fact, error) {
	seen := map[string]bool{id: true}
	frontier := []string{id}
	var out []*Fact
	for d := 0; d < depth; d++ {
		var next []string
		for _, fid := range frontier {
			rows, err := s.db.Query(`SELECT to_id FROM links WHERE from_id=? UNION SELECT from_id FROM links WHERE to_id=?`, fid, fid)
			if err != nil {
				return nil, err
			}
			var ids []string
			for rows.Next() {
				var nid string
				rows.Scan(&nid)
				if !seen[nid] {
					seen[nid] = true
					ids = append(ids, nid)
				}
			}
			rows.Close() // close before nested Gets: the pool holds one conn
			for _, nid := range ids {
				next = append(next, nid)
				if f, err := s.Get(nid); err == nil && f.Status != StatusRetracted {
					out = append(out, f)
				}
			}
		}
		frontier = next
	}
	return out, nil
}

func (s *Store) Links(id string) (map[LinkType][]string, error) {
	rows, err := s.db.Query(`SELECT to_id, type FROM links WHERE from_id=? UNION SELECT from_id, type FROM links WHERE to_id=?`, id, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[LinkType][]string{}
	for rows.Next() {
		var oid, t string
		rows.Scan(&oid, &t)
		out[LinkType(t)] = append(out[LinkType(t)], oid)
	}
	return out, nil
}

// SetEmbedding stores a fact's embedding vector (little-endian float32 blob).
func (s *Store) SetEmbedding(id string, vec []float32) error {
	buf := make([]byte, 4*len(vec))
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	res, err := s.db.Exec(`UPDATE facts SET embedding=? WHERE id=?`, buf, id)
	return execErr(res, err, id)
}

// Embeddings returns id -> vector for non-retracted facts visible to the
// session (VERIFIED project-wide + own PROPOSED; "" = all).
// Brute-force cosine over this map is the v0 nearest-neighbor path (a session
// store holds tens-to-hundreds of facts; sqlite-vec deferred until size demands).
func (s *Store) Embeddings(sessionID string) (map[string][]float32, error) {
	rows, err := s.db.Query(`SELECT id, embedding FROM facts WHERE status!=? AND embedding IS NOT NULL`+sessionScope,
		string(StatusRetracted), string(StatusVerified), sessionID, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string][]float32{}
	for rows.Next() {
		var id string
		var buf []byte
		if err := rows.Scan(&id, &buf); err != nil {
			return nil, err
		}
		vec := make([]float32, len(buf)/4)
		for i := range vec {
			vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
		out[id] = vec
	}
	return out, rows.Err()
}

// Audit checks the deterministic invariants (spec §5) — CI / debug command.
func (s *Store) Audit() []string {
	var issues []string
	// inv 1 (shape only: spans exist and are well-formed; byte resolution needs the transcript)
	rows, _ := s.db.Query(`SELECT f.id FROM facts f LEFT JOIN provenance p ON p.fact_id=f.id WHERE p.fact_id IS NULL`)
	for rows.Next() {
		var id string
		rows.Scan(&id)
		issues = append(issues, "no provenance: "+id)
	}
	rows.Close()
	// inv 2: DERIVED parent chains bottom out in non-DERIVED, no cycles
	rows, _ = s.db.Query(`SELECT id FROM facts WHERE kind=? AND status!=?`, string(KindDerived), string(StatusRetracted))
	var derived []string
	for rows.Next() {
		var id string
		rows.Scan(&id)
		derived = append(derived, id)
	}
	rows.Close()
	for _, id := range derived {
		if !s.grounded(id, map[string]bool{}) {
			issues = append(issues, "ungrounded DERIVED: "+id)
		}
	}
	// inv 3: no live CONTRADICTS between two VERIFIED non-stale facts
	rows, _ = s.db.Query(`SELECT l.from_id, l.to_id FROM links l
		JOIN facts a ON a.id=l.from_id JOIN facts b ON b.id=l.to_id
		WHERE l.type=? AND a.status=? AND b.status=? AND a.stale=0 AND b.stale=0`,
		string(LinkContradicts), string(StatusVerified), string(StatusVerified))
	for rows.Next() {
		var a, b string
		rows.Scan(&a, &b)
		issues = append(issues, fmt.Sprintf("live contradiction in core: %s vs %s", a, b))
	}
	rows.Close()
	return issues
}

func (s *Store) grounded(id string, visiting map[string]bool) bool {
	if visiting[id] {
		return false // cycle
	}
	visiting[id] = true
	defer delete(visiting, id)
	f, err := s.Get(id)
	if err != nil {
		return false
	}
	if f.Kind != KindDerived {
		return true
	}
	if len(f.Parents) == 0 {
		return false
	}
	for _, p := range f.Parents {
		if !s.grounded(p, visiting) {
			return false
		}
	}
	return true
}

// ---- scan helpers ----

type rowScanner interface{ Scan(dest ...any) error }

func scanFact(r rowScanner) (*Fact, error) {
	var f Fact
	var kind, status, trust, ents, pars, created, confirmed, rts string
	var stale, pinned int
	if err := r.Scan(&f.ID, &f.Statement, &kind, &status, &stale, &pinned, &trust, &f.Confidence,
		&ents, &pars, &f.SessionID, &created, &confirmed, &rts); err != nil {
		return nil, err
	}
	f.Kind, f.Status, f.Trust = Kind(kind), Status(status), Trust(trust)
	f.Stale, f.Pinned = stale == 1, pinned == 1
	json.Unmarshal([]byte(ents), &f.Entities)
	json.Unmarshal([]byte(pars), &f.Parents)
	json.Unmarshal([]byte(rts), &f.RenderTurns)
	f.Created, _ = time.Parse(time.RFC3339, created)
	f.LastConfirmed, _ = time.Parse(time.RFC3339, confirmed)
	return &f, nil
}

func scanFacts(rows *sql.Rows) ([]*Fact, error) {
	var out []*Fact
	for rows.Next() {
		f, err := scanFact(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, rows.Err()
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func execErr(res sql.Result, err error, id string) error {
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return fmt.Errorf("fact not found or not updatable: %s", id)
	}
	return nil
}
