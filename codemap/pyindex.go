// Package codemap is an in-harness symbol server: a deterministic structural
// index of the code on disk (symbols, signatures, imports, references) that a
// harnessed model queries in tens of tokens instead of re-reading files in
// thousands, with a drift report after every mutation.
//
// Principles (transplanted from ctxmap, see ~/codemap-spec.md):
//   - The index is a CACHE; the disk is ground truth. Rebuildable, never
//     load-bearing; indexer death degrades to plain file reads.
//   - NO model in the loop: everything served is parser output. The
//     extraction-quality failure mode of the fact path cannot exist here.
//   - Tiny answers, with code_body as the exact-span escalation.
//
// v0 backend: Python via a stdlib-ast subprocess — deliberately boring; no
// LSP lifecycle, no cgo. The Indexer seam is where gopls/tree-sitter land.
package codemap

import (
	"encoding/json"
	"fmt"
	"os/exec"
)

// Symbol is one indexed definition.
type Symbol struct {
	Kind      string `json:"kind"`   // function | class | method
	Name      string `json:"name"`   // bare name
	Parent    string `json:"parent"` // class name for methods, "" otherwise
	Signature string `json:"sig"`
	Line      int    `json:"line"`
	End       int    `json:"end"`
	Doc       string `json:"doc"` // first docstring line, if any
}

// Import is one import statement.
type Import struct {
	Module string   `json:"module"` // "" for plain `import x` entries (see Names)
	Names  []string `json:"names"`
	Level  int      `json:"level"` // relative-import level (from . import x => 1)
	Line   int      `json:"line"`
}

// FileIndex is the structural projection of one file.
type FileIndex struct {
	Symbols []Symbol         `json:"symbols"`
	Imports []Import         `json:"imports"`
	Refs    map[string][]int `json:"refs"`  // name -> lines where referenced
	Error   string           `json:"error"` // parse error, if any
}

// Indexer is the backend seam (v0: python-ast; later: LSP, tree-sitter).
type Indexer interface {
	// Index parses the given repo-relative files (all indexable files under
	// root when rels is empty) and returns their projections.
	Index(root string, rels []string) (map[string]*FileIndex, error)
}

// pyScript is the whole v0 backend: stdlib-only, JSON out. Symbols with
// header-accurate signatures (paren-balanced header scan), imports with
// relative level, and name/attribute reference lines.
const pyScript = `
import ast, json, os, sys

root = sys.argv[1]
targets = sys.argv[2:]

def header_sig(src_lines, node):
    out, depth, started = [], 0, False
    for i in range(node.lineno - 1, min(node.lineno + 20, len(src_lines))):
        line = src_lines[i]
        out.append(line.strip())
        for ch in line:
            if ch in "([{": depth += 1; started = True
            elif ch in ")]}": depth -= 1
        if started and depth <= 0 and line.rstrip().endswith(":"):
            break
        if not started and line.rstrip().endswith(":"):
            break
    sig = " ".join(out)
    return sig[:-1].strip() if sig.endswith(":") else sig.strip()

def index_file(rel):
    p = os.path.join(root, rel)
    try:
        src = open(p, encoding="utf-8", errors="replace").read()
    except OSError as e:
        return {"symbols": [], "imports": [], "refs": {}, "error": str(e)}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"symbols": [], "imports": [], "refs": {},
                "error": "syntax error at line %s: %s" % (e.lineno, e.msg)}
    lines = src.splitlines()
    syms, imports, refs = [], [], {}

    def doc1(node):
        d = ast.get_docstring(node) or ""
        return d.splitlines()[0][:120] if d else ""

    def add(node, kind, parent=""):
        syms.append({"kind": kind, "name": node.name, "parent": parent,
                     "sig": header_sig(lines, node), "line": node.lineno,
                     "end": getattr(node, "end_lineno", node.lineno),
                     "doc": doc1(node)})

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(n, "function")
        elif isinstance(n, ast.ClassDef):
            add(n, "class")
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(m, "method", n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            # module-level constants: MAGIC = b"CWL0", VERSION = 2, ...
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tgts:
                if isinstance(t, ast.Name):
                    syms.append({"kind": "const", "name": t.id, "parent": "",
                                 "sig": (lines[n.lineno - 1] or "").strip(),
                                 "line": n.lineno,
                                 "end": getattr(n, "end_lineno", n.lineno),
                                 "doc": ""})

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports.append({"module": "", "names": [a.name for a in n.names],
                            "level": 0, "line": n.lineno})
        elif isinstance(n, ast.ImportFrom):
            imports.append({"module": n.module or "", "names": [a.name for a in n.names],
                            "level": n.level, "line": n.lineno})
        elif isinstance(n, ast.Name):
            refs.setdefault(n.id, []).append(n.lineno)
        elif isinstance(n, ast.Attribute):
            refs.setdefault(n.attr, []).append(n.lineno)

    return {"symbols": syms, "imports": imports, "refs": refs, "error": ""}

if not targets:
    targets = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", "node_modules")]
        for f in filenames:
            if f.endswith(".py"):
                targets.append(os.path.relpath(os.path.join(dirpath, f), root))

print(json.dumps({rel: index_file(rel) for rel in sorted(targets)}))
`

// PyIndexer indexes Python files via a python3 subprocess.
type PyIndexer struct{}

func (PyIndexer) Index(root string, rels []string) (map[string]*FileIndex, error) {
	args := append([]string{"-c", pyScript, root}, rels...)
	out, err := exec.Command("python3", args...).Output()
	if err != nil {
		return nil, fmt.Errorf("codemap: python indexer: %w", err)
	}
	var m map[string]*FileIndex
	if err := json.Unmarshal(out, &m); err != nil {
		return nil, fmt.Errorf("codemap: bad indexer output: %w", err)
	}
	return m, nil
}
