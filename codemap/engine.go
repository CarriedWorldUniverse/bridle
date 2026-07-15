package codemap

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Engine holds the per-repo index and answers structure queries. All answers
// are budgeted-small; Body is the exact-span escalation. Fail-open: a dead
// indexer yields "index unavailable" answers, never an error that stops a turn.
type Engine struct {
	root string
	ix   Indexer

	mu     sync.Mutex
	files  map[string]*FileIndex
	mtimes map[string]time.Time
}

// New builds an engine rooted at dir and performs the initial full index.
// Indexing errors are non-fatal (the engine starts empty and lazily retries).
func New(dir string, ix Indexer) *Engine {
	e := &Engine{root: dir, ix: ix, files: map[string]*FileIndex{}, mtimes: map[string]time.Time{}}
	if m, err := ix.Index(dir, nil); err == nil {
		e.files = m
		e.stampMtimes()
	}
	return e
}

func (e *Engine) stampMtimes() {
	for rel := range e.files {
		if fi, err := os.Stat(filepath.Join(e.root, rel)); err == nil {
			e.mtimes[rel] = fi.ModTime()
		}
	}
}

// Reindex re-parses one repo-relative file and returns the drift report:
// human-readable lines describing what no longer matches (changed/removed
// signatures with their cross-file callers, new parse errors). Empty = clean.
func (e *Engine) Reindex(rel string) []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	old := e.files[rel]
	m, err := e.ix.Index(e.root, []string{rel})
	if err != nil || m[rel] == nil {
		return nil // fail open — stale index beats a broken turn
	}
	e.files[rel] = m[rel]
	if fi, err := os.Stat(filepath.Join(e.root, rel)); err == nil {
		e.mtimes[rel] = fi.ModTime()
	}
	return e.driftLocked(rel, old, m[rel])
}

// driftLocked compares old->new for one file. Caller holds e.mu.
func (e *Engine) driftLocked(rel string, old, cur *FileIndex) []string {
	var out []string
	if cur.Error != "" {
		out = append(out, fmt.Sprintf("%s: %s", rel, cur.Error))
	}
	if old == nil {
		return out
	}
	oldSigs := map[string]string{}
	for _, s := range old.Symbols {
		oldSigs[s.Parent+"."+s.Name] = s.Signature
	}
	curSigs := map[string]string{}
	for _, s := range cur.Symbols {
		curSigs[s.Parent+"."+s.Name] = s.Signature
	}
	for key, oldSig := range oldSigs {
		name := key[strings.LastIndex(key, ".")+1:]
		curSig, ok := curSigs[key]
		switch {
		case !ok:
			out = append(out, fmt.Sprintf("%s: %s REMOVED%s", rel, name, e.callersNote(rel, name)))
		case curSig != oldSig:
			out = append(out, fmt.Sprintf("%s: %s signature changed: %q -> %q%s", rel, name, oldSig, curSig, e.callersNote(rel, name)))
		}
	}
	return out
}

// callersNote lists other files referencing name — the blast radius of a
// removal or signature change. Caller holds e.mu.
func (e *Engine) callersNote(exceptRel, name string) string {
	var sites []string
	for rel, fi := range e.files {
		if rel == exceptRel || fi == nil {
			continue
		}
		if lines := fi.Refs[name]; len(lines) > 0 {
			sites = append(sites, fmt.Sprintf("%s:%d", rel, lines[0]))
		}
	}
	if len(sites) == 0 {
		return ""
	}
	sort.Strings(sites)
	return " (referenced by " + strings.Join(sites, ", ") + ")"
}

// SweepChanged lazily re-indexes any file whose mtime moved (run_command can
// mutate files: formatters, codegen, git). Returns accumulated drift.
func (e *Engine) SweepChanged() []string {
	e.mu.Lock()
	var stale []string
	for rel, prev := range e.mtimes {
		if fi, err := os.Stat(filepath.Join(e.root, rel)); err == nil && !fi.ModTime().Equal(prev) {
			stale = append(stale, rel)
		}
	}
	e.mu.Unlock()
	var out []string
	for _, rel := range stale {
		out = append(out, e.Reindex(rel)...)
	}
	return out
}

// ---- queries (each answer deliberately tiny) ----

const unavailable = "codemap index unavailable — read the file directly"

// Outline lists a file's symbols: kind, name, signature, line span.
func (e *Engine) Outline(rel string) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	fi := e.files[rel]
	if fi == nil {
		return "no index for " + rel + " (not a Python file, or not indexed)"
	}
	if fi.Error != "" {
		return rel + ": " + fi.Error
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s — %d symbols:\n", rel, len(fi.Symbols))
	for _, s := range fi.Symbols {
		indent := ""
		if s.Parent != "" {
			indent = "  "
		}
		fmt.Fprintf(&b, "%s%s %s  [L%d-%d]", indent, s.Kind, s.Signature, s.Line, s.End)
		if s.Doc != "" {
			fmt.Fprintf(&b, "  # %s", s.Doc)
		}
		b.WriteString("\n")
	}
	for _, im := range fi.Imports {
		fmt.Fprintf(&b, "import: from %s%s import %s  [L%d]\n", strings.Repeat(".", im.Level), im.Module, strings.Join(im.Names, ", "), im.Line)
	}
	return strings.TrimRight(b.String(), "\n")
}

// Symbol finds a definition by name (exact, then substring) across the repo.
func (e *Engine) Symbol(name string) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.files) == 0 {
		return unavailable
	}
	var exact, fuzzy []string
	for _, rel := range e.sortedFiles() {
		for _, s := range e.files[rel].Symbols {
			line := fmt.Sprintf("%s:%d  %s %s", rel, s.Line, s.Kind, s.Signature)
			if s.Doc != "" {
				line += "  # " + s.Doc
			}
			if s.Name == name {
				exact = append(exact, line)
			} else if strings.Contains(strings.ToLower(s.Name), strings.ToLower(name)) {
				fuzzy = append(fuzzy, line)
			}
		}
	}
	if len(exact) > 0 {
		return strings.Join(exact, "\n")
	}
	if len(fuzzy) > 0 {
		return "no exact match; similar:\n" + strings.Join(cap10(fuzzy), "\n")
	}
	return "no symbol named " + name
}

// Refs lists where a name is referenced, per file with line numbers.
func (e *Engine) Refs(name string) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.files) == 0 {
		return unavailable
	}
	var out []string
	for _, rel := range e.sortedFiles() {
		if lines := e.files[rel].Refs[name]; len(lines) > 0 {
			out = append(out, fmt.Sprintf("%s: lines %s", rel, intsToStr(cap12(lines))))
		}
	}
	if len(out) == 0 {
		return "no references to " + name
	}
	return strings.Join(out, "\n")
}

// Body returns the exact source span of a named symbol — the escalation path
// when the projection isn't enough.
func (e *Engine) Body(name string) string {
	e.mu.Lock()
	var rel string
	var sym *Symbol
	for _, r := range e.sortedFiles() {
		for i, s := range e.files[r].Symbols {
			if s.Name == name {
				rel, sym = r, &e.files[r].Symbols[i]
				break
			}
		}
		if sym != nil {
			break
		}
	}
	e.mu.Unlock()
	if sym == nil {
		return "no symbol named " + name
	}
	raw, err := os.ReadFile(filepath.Join(e.root, rel))
	if err != nil {
		return "read failed: " + err.Error()
	}
	lines := strings.Split(string(raw), "\n")
	lo, hi := sym.Line-1, sym.End
	if lo < 0 {
		lo = 0
	}
	if hi > len(lines) {
		hi = len(lines)
	}
	return fmt.Sprintf("%s:%d-%d\n%s", rel, sym.Line, sym.End, strings.Join(lines[lo:hi], "\n"))
}

// Diag reports current inconsistencies: parse errors and intra-repo imports
// naming symbols that don't exist (approximate by design in v0).
func (e *Engine) Diag() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.files) == 0 {
		return unavailable
	}
	var out []string
	// symbol universe (top-level names per module basename)
	modSyms := map[string]map[string]bool{}
	for rel, fi := range e.files {
		mod := strings.TrimSuffix(filepath.Base(rel), ".py")
		if modSyms[mod] == nil {
			modSyms[mod] = map[string]bool{}
		}
		for _, s := range fi.Symbols {
			if s.Parent == "" {
				modSyms[mod][s.Name] = true
			}
		}
	}
	for _, rel := range e.sortedFiles() {
		fi := e.files[rel]
		if fi.Error != "" {
			out = append(out, fmt.Sprintf("%s: %s", rel, fi.Error))
		}
		for _, im := range fi.Imports {
			mod := im.Module
			if i := strings.LastIndex(mod, "."); i >= 0 {
				mod = mod[i+1:]
			}
			syms, known := modSyms[mod]
			if !known || im.Level == 0 && im.Module == "" {
				continue // external or plain `import x` — out of v0 scope
			}
			for _, n := range im.Names {
				if n != "*" && !syms[n] {
					out = append(out, fmt.Sprintf("%s:%d: import %q from %s does not resolve (no such top-level symbol)", rel, im.Line, n, im.Module))
				}
			}
		}
	}
	if len(out) == 0 {
		return "no inconsistencies found"
	}
	return strings.Join(out, "\n")
}

func (e *Engine) sortedFiles() []string {
	rels := make([]string, 0, len(e.files))
	for rel := range e.files {
		rels = append(rels, rel)
	}
	sort.Strings(rels)
	return rels
}

func cap10(s []string) []string {
	if len(s) > 10 {
		return s[:10]
	}
	return s
}

func cap12(l []int) []int {
	if len(l) > 12 {
		return l[:12]
	}
	return l
}

func intsToStr(l []int) string {
	parts := make([]string, len(l))
	for i, n := range l {
		parts[i] = fmt.Sprint(n)
	}
	return strings.Join(parts, ",")
}
