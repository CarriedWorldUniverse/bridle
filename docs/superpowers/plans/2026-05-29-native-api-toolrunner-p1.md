# Native-API ToolRunner — P1 (LocalToolRunner) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `LocalToolRunner` — an all-Go `bridle.ToolRunner` that gives a native-API aspect the local coding-agent tool suite (bash / read / write / edit / glob / grep / web_fetch / web_extract), and prove a real multi-step tool chain end-to-end against DeepSeek via bridle's openai provider.

**Architecture:** A new reusable package `github.com/CarriedWorldUniverse/bridle/toolrunner` (sibling to `fake/`, `stubfunnel/`). `LocalToolRunner` implements `bridle.ToolRunner.Run` by routing on `call.Name` into per-tool handlers, all executing in-process. It also exposes `Defs()` → `[]bridle.ToolDef` (the JSON-schema tool surface). The funnel (`nexus/runtime/cmd/agentfunnel`) imports it; lane-2 host tools and the permission model are later phases (P2/P3) and out of scope here. No bridle harness changes.

**Tech Stack:** Go, stdlib (`os/exec`, `os`, `path/filepath`, `regexp`, `net/http`, `encoding/json`), bridle (`Harness`, `TurnRequest`, `ToolRunner`, `ToolDef`, provider/openai). DeepSeek (OpenAI-compatible `/v1`, model `deepseek-chat`) for the live chain.

**Placement decision (flag):** the design (§4) calls the runner "funnel-owned". P1's local lane has no nexus dependency, so it lives in bridle as a generic importable package — DRY, testable standalone, and the funnel just imports it. If the operator prefers it inside `nexus/runtime`, the package moves wholesale (self-contained). Lane 2 (P2) genuinely needs the broker and will live in the funnel, importing this router.

**Safety note:** P1 has **no permission gating** (that's P3). `bash`/`write`/`edit` execute as the host user. The live e2e test runs every tool inside a fresh `t.TempDir()` workdir with a tightly-scoped prompt — acceptable for a proving build, NOT production posture.

---

## File Structure

All under `~/Source/bridle/toolrunner/`:

- `config.go` — `Config` (WorkDir, Lynxai endpoint/key, HTTPClient, BashTimeout) + `New(Config) *LocalToolRunner`.
- `local.go` — `LocalToolRunner` struct + `Run` router (switch on `call.Name`) + arg-decode helper + `Defs()`.
- `bash.go` — `bash` handler.
- `files.go` — `read` / `write` / `edit` handlers + safe path resolution.
- `search.go` — `glob` / `grep` handlers.
- `web.go` — `web_fetch` / `web_extract` (lynxai HTTP client).
- `defs.go` — the `[]bridle.ToolDef` schemas.
- `*_test.go` — per-file unit tests (table-driven, in-process, no network except web via `httptest`).
- `e2e_test.go` — env-gated live DeepSeek chain (skips without `BRIDLE_E2E_OPENAI_KEY`).

Branch: `feat/toolrunner-p1-local` off `main`.

---

## Conventions (apply in every handler)

- **Args:** each tool's args is a Go struct; decode with `json.Unmarshal(call.Args, &a)`; on bad JSON return `(nil, fmt.Errorf("tool %s: bad args: %w", call.Name, err))`.
- **Tool-level failures** (command non-zero, file missing) are returned as a **successful** `json.RawMessage` result that *describes* the failure (so the model sees and reacts), NOT a Go `error`. Reserve the Go `error` return for harness-level faults (bad args, unknown tool, runner misconfig). This matches design §12 ("tool failures return as tool_result errors so the model can react").
- **Path safety:** all file paths resolve relative to `WorkDir`; reject paths that escape it (`..`). Helper `resolve(path) (string, error)` in `files.go`.
- **JSON results:** marshal a small struct; never return raw unbounded blobs without a cap (truncate stdout/file reads to a sane max, note truncation).

---

## Task 0: Scaffold package + Config + skeleton compile

**Files:**
- Create: `toolrunner/config.go`
- Create: `toolrunner/local.go`

- [ ] **Step 1: Branch**

```bash
cd ~/Source/bridle && git checkout main && git pull --ff-only && git checkout -b feat/toolrunner-p1-local
```

- [ ] **Step 2: `config.go`**

```go
package toolrunner

import (
	"net/http"
	"time"
)

// Config configures a LocalToolRunner. Zero values are filled with
// sane defaults by New.
type Config struct {
	// WorkDir is the root for relative file paths and the cwd for bash.
	// Required; New returns an error if empty.
	WorkDir string

	// LynxaiBaseURL is the base URL of the shared lynxai service
	// (e.g. "http://127.0.0.1:7878"). Empty disables web_fetch/web_extract
	// (they return a tool-level error telling the model web access is off).
	LynxaiBaseURL string
	// LynxaiKey is an optional bearer token for lynxai (reverse-proxy auth).
	LynxaiKey string

	// HTTPClient is used for lynxai calls. Nil → a client with a 60s timeout.
	HTTPClient *http.Client

	// BashTimeout caps a single bash command. Zero → 120s.
	BashTimeout time.Duration

	// MaxOutputBytes caps captured stdout/stderr and file reads. Zero → 1<<20 (1 MiB).
	MaxOutputBytes int
}
```

- [ ] **Step 3: `local.go` skeleton**

```go
package toolrunner

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// LocalToolRunner executes the local coding-agent tool suite in-process.
// It implements bridle.ToolRunner.
type LocalToolRunner struct {
	cfg Config
}

// New builds a LocalToolRunner, filling defaults. Returns an error if
// WorkDir is empty.
func New(cfg Config) (*LocalToolRunner, error) {
	if cfg.WorkDir == "" {
		return nil, fmt.Errorf("toolrunner: WorkDir is required")
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 60 * time.Second}
	}
	if cfg.BashTimeout == 0 {
		cfg.BashTimeout = 120 * time.Second
	}
	if cfg.MaxOutputBytes == 0 {
		cfg.MaxOutputBytes = 1 << 20
	}
	return &LocalToolRunner{cfg: cfg}, nil
}

// Run routes a tool call to its handler. Unknown tools return an error.
func (r *LocalToolRunner) Run(ctx context.Context, call bridle.ToolCall) (json.RawMessage, error) {
	switch call.Name {
	case "bash":
		return r.runBash(ctx, call.Args)
	case "read":
		return r.runRead(call.Args)
	case "write":
		return r.runWrite(call.Args)
	case "edit":
		return r.runEdit(call.Args)
	case "glob":
		return r.runGlob(call.Args)
	case "grep":
		return r.runGrep(call.Args)
	case "web_fetch":
		return r.runWebFetch(ctx, call.Args)
	case "web_extract":
		return r.runWebExtract(ctx, call.Args)
	default:
		return nil, fmt.Errorf("toolrunner: unknown tool %q", call.Name)
	}
}

// result marshals v to json.RawMessage, converting marshal failure to a Go error.
func result(v any) (json.RawMessage, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("toolrunner: marshal result: %w", err)
	}
	return b, nil
}
```

- [ ] **Step 4: Stub the eight handlers so it compiles** — add to `local.go` temporarily (each replaced in its task):

```go
func (r *LocalToolRunner) runBash(ctx context.Context, args json.RawMessage) (json.RawMessage, error) { return nil, fmt.Errorf("todo") }
func (r *LocalToolRunner) runRead(args json.RawMessage) (json.RawMessage, error)  { return nil, fmt.Errorf("todo") }
func (r *LocalToolRunner) runWrite(args json.RawMessage) (json.RawMessage, error) { return nil, fmt.Errorf("todo") }
func (r *LocalToolRunner) runEdit(args json.RawMessage) (json.RawMessage, error)  { return nil, fmt.Errorf("todo") }
func (r *LocalToolRunner) runGlob(args json.RawMessage) (json.RawMessage, error)  { return nil, fmt.Errorf("todo") }
func (r *LocalToolRunner) runGrep(args json.RawMessage) (json.RawMessage, error)  { return nil, fmt.Errorf("todo") }
func (r *LocalToolRunner) runWebFetch(ctx context.Context, args json.RawMessage) (json.RawMessage, error)    { return nil, fmt.Errorf("todo") }
func (r *LocalToolRunner) runWebExtract(ctx context.Context, args json.RawMessage) (json.RawMessage, error)  { return nil, fmt.Errorf("todo") }
```

- [ ] **Step 5: Compile** — `cd ~/Source/bridle && go build ./toolrunner/` → expect success.
- [ ] **Step 6: Commit** — `git add toolrunner/ docs/superpowers/plans/ && git commit -m "feat(toolrunner): scaffold LocalToolRunner package + config"`

---

## Task 1: `bash` tool (TDD)

**Files:** Create `toolrunner/bash.go`, `toolrunner/bash_test.go`. Replace the `runBash` stub.

- [ ] **Step 1: Failing test** — `bash_test.go`:

```go
package toolrunner

import (
	"context"
	"encoding/json"
	"testing"
)

func newTestRunner(t *testing.T) *LocalToolRunner {
	t.Helper()
	r, err := New(Config{WorkDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestBashEcho(t *testing.T) {
	r := newTestRunner(t)
	out, err := r.runBash(context.Background(), json.RawMessage(`{"command":"echo hello"}`))
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	var res bashResult
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 0 || res.Stdout != "hello\n" {
		t.Fatalf("got %+v", res)
	}
}

func TestBashNonZeroIsToolResultNotError(t *testing.T) {
	r := newTestRunner(t)
	out, err := r.runBash(context.Background(), json.RawMessage(`{"command":"exit 3"}`))
	if err != nil {
		t.Fatalf("non-zero exit must be a tool result, not a Go error: %v", err)
	}
	var res bashResult
	_ = json.Unmarshal(out, &res)
	if res.ExitCode != 3 {
		t.Fatalf("want exit 3, got %+v", res)
	}
}
```

- [ ] **Step 2: Run, expect FAIL** — `go test ./toolrunner/ -run TestBash -v` → undefined `bashResult` / stub error.

- [ ] **Step 3: Implement `bash.go`**

```go
package toolrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"time"
)

type bashArgs struct {
	Command   string `json:"command"`
	TimeoutMS int    `json:"timeout_ms,omitempty"`
}

type bashResult struct {
	Stdout    string `json:"stdout"`
	Stderr    string `json:"stderr"`
	ExitCode  int    `json:"exit_code"`
	TimedOut  bool   `json:"timed_out,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

func (r *LocalToolRunner) runBash(ctx context.Context, raw json.RawMessage) (json.RawMessage, error) {
	var a bashArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("bash: bad args: %w", err)
	}
	if a.Command == "" {
		return result(bashResult{Stderr: "bash: empty command", ExitCode: -1})
	}
	timeout := r.cfg.BashTimeout
	if a.TimeoutMS > 0 {
		timeout = time.Duration(a.TimeoutMS) * time.Millisecond
	}
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cctx, "bash", "-c", a.Command)
	cmd.Dir = r.cfg.WorkDir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	res := bashResult{}
	res.Stdout, res.Truncated = capString(stdout.String(), r.cfg.MaxOutputBytes)
	s2, t2 := capString(stderr.String(), r.cfg.MaxOutputBytes)
	res.Stderr = s2
	res.Truncated = res.Truncated || t2

	if cctx.Err() == context.DeadlineExceeded {
		res.TimedOut = true
		res.ExitCode = -1
		return result(res)
	}
	if runErr != nil {
		var ee *exec.ExitError
		if errors.As(runErr, &ee) {
			res.ExitCode = ee.ExitCode()
			return result(res)
		}
		// couldn't even start the process → harness-level error
		return nil, fmt.Errorf("bash: %w", runErr)
	}
	res.ExitCode = 0
	return result(res)
}

// capString truncates s to max bytes, returning (s', truncated).
func capString(s string, max int) (string, bool) {
	if max > 0 && len(s) > max {
		return s[:max], true
	}
	return s, false
}
```

- [ ] **Step 4: Run, expect PASS** — `go test ./toolrunner/ -run TestBash -v`.
- [ ] **Step 5: Commit** — `git commit -am "feat(toolrunner): bash tool (per-user exec, non-zero as tool result)"`

---

## Task 2: `read` / `write` / `edit` + path safety (TDD)

**Files:** Create `toolrunner/files.go`, `toolrunner/files_test.go`. Replace the three stubs.

- [ ] **Step 1: Failing test** — `files_test.go`:

```go
package toolrunner

import (
	"encoding/json"
	"testing"
)

func TestWriteThenRead(t *testing.T) {
	r := newTestRunner(t)
	if _, err := r.runWrite(json.RawMessage(`{"path":"a.txt","content":"hi there"}`)); err != nil {
		t.Fatal(err)
	}
	out, err := r.runRead(json.RawMessage(`{"path":"a.txt"}`))
	if err != nil {
		t.Fatal(err)
	}
	var res readResult
	_ = json.Unmarshal(out, &res)
	if res.Content != "hi there" {
		t.Fatalf("got %q", res.Content)
	}
}

func TestEditReplace(t *testing.T) {
	r := newTestRunner(t)
	_, _ = r.runWrite(json.RawMessage(`{"path":"b.txt","content":"foo bar foo"}`))
	out, err := r.runEdit(json.RawMessage(`{"path":"b.txt","old_string":"foo","new_string":"baz","replace_all":true}`))
	if err != nil {
		t.Fatal(err)
	}
	var res editResult
	_ = json.Unmarshal(out, &res)
	if res.Replacements != 2 {
		t.Fatalf("want 2 replacements, got %+v", res)
	}
	rd, _ := r.runRead(json.RawMessage(`{"path":"b.txt"}`))
	var got readResult
	_ = json.Unmarshal(rd, &got)
	if got.Content != "baz bar baz" {
		t.Fatalf("got %q", got.Content)
	}
}

func TestPathEscapeRejected(t *testing.T) {
	r := newTestRunner(t)
	out, err := r.runRead(json.RawMessage(`{"path":"../../etc/passwd"}`))
	if err != nil {
		t.Fatalf("escape should be a tool result, not a Go error: %v", err)
	}
	var res readResult
	_ = json.Unmarshal(out, &res)
	if res.Error == "" {
		t.Fatalf("expected an error field for escaping path, got %+v", res)
	}
}
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `files.go`**

```go
package toolrunner

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// resolve joins p onto WorkDir and rejects any path escaping WorkDir.
func (r *LocalToolRunner) resolve(p string) (string, error) {
	if p == "" {
		return "", fmt.Errorf("empty path")
	}
	root := filepath.Clean(r.cfg.WorkDir)
	full := filepath.Clean(filepath.Join(root, p))
	if full != root && !strings.HasPrefix(full, root+string(os.PathSeparator)) {
		return "", fmt.Errorf("path %q escapes workdir", p)
	}
	return full, nil
}

type readArgs struct {
	Path string `json:"path"`
}
type readResult struct {
	Content   string `json:"content,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
	Error     string `json:"error,omitempty"`
}

func (r *LocalToolRunner) runRead(raw json.RawMessage) (json.RawMessage, error) {
	var a readArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("read: bad args: %w", err)
	}
	full, err := r.resolve(a.Path)
	if err != nil {
		return result(readResult{Error: err.Error()})
	}
	b, err := os.ReadFile(full)
	if err != nil {
		return result(readResult{Error: err.Error()})
	}
	s, trunc := capString(string(b), r.cfg.MaxOutputBytes)
	return result(readResult{Content: s, Truncated: trunc})
}

type writeArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}
type writeResult struct {
	BytesWritten int    `json:"bytes_written,omitempty"`
	Error        string `json:"error,omitempty"`
}

func (r *LocalToolRunner) runWrite(raw json.RawMessage) (json.RawMessage, error) {
	var a writeArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("write: bad args: %w", err)
	}
	full, err := r.resolve(a.Path)
	if err != nil {
		return result(writeResult{Error: err.Error()})
	}
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return result(writeResult{Error: err.Error()})
	}
	if err := os.WriteFile(full, []byte(a.Content), 0o644); err != nil {
		return result(writeResult{Error: err.Error()})
	}
	return result(writeResult{BytesWritten: len(a.Content)})
}

type editArgs struct {
	Path       string `json:"path"`
	OldString  string `json:"old_string"`
	NewString  string `json:"new_string"`
	ReplaceAll bool   `json:"replace_all,omitempty"`
}
type editResult struct {
	Replacements int    `json:"replacements,omitempty"`
	Error        string `json:"error,omitempty"`
}

func (r *LocalToolRunner) runEdit(raw json.RawMessage) (json.RawMessage, error) {
	var a editArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("edit: bad args: %w", err)
	}
	full, err := r.resolve(a.Path)
	if err != nil {
		return result(editResult{Error: err.Error()})
	}
	b, err := os.ReadFile(full)
	if err != nil {
		return result(editResult{Error: err.Error()})
	}
	content := string(b)
	n := strings.Count(content, a.OldString)
	if a.OldString == "" || n == 0 {
		return result(editResult{Error: "old_string not found"})
	}
	if !a.ReplaceAll && n > 1 {
		return result(editResult{Error: fmt.Sprintf("old_string is not unique (%d matches); set replace_all or add context", n)})
	}
	if a.ReplaceAll {
		content = strings.ReplaceAll(content, a.OldString, a.NewString)
	} else {
		content = strings.Replace(content, a.OldString, a.NewString, 1)
		n = 1
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		return result(editResult{Error: err.Error()})
	}
	return result(editResult{Replacements: n})
}
```

- [ ] **Step 4: Run, expect PASS** — `go test ./toolrunner/ -run 'TestWrite|TestEdit|TestPath' -v`.
- [ ] **Step 5: Commit** — `git commit -am "feat(toolrunner): read/write/edit file tools + workdir path safety"`

---

## Task 3: `glob` / `grep` (TDD)

**Files:** Create `toolrunner/search.go`, `toolrunner/search_test.go`. Replace the two stubs.

- [ ] **Step 1: Failing test** — `search_test.go`:

```go
package toolrunner

import (
	"encoding/json"
	"testing"
)

func TestGlobAndGrep(t *testing.T) {
	r := newTestRunner(t)
	_, _ = r.runWrite(json.RawMessage(`{"path":"x/one.go","content":"package x\n// TODO: hi\n"}`))
	_, _ = r.runWrite(json.RawMessage(`{"path":"x/two.txt","content":"nothing here"}`))

	g, err := r.runGlob(json.RawMessage(`{"pattern":"**/*.go"}`))
	if err != nil {
		t.Fatal(err)
	}
	var gr globResult
	_ = json.Unmarshal(g, &gr)
	if len(gr.Matches) != 1 || gr.Matches[0] != "x/one.go" {
		t.Fatalf("glob got %+v", gr.Matches)
	}

	gp, err := r.runGrep(json.RawMessage(`{"pattern":"TODO","glob":"**/*.go"}`))
	if err != nil {
		t.Fatal(err)
	}
	var gpr grepResult
	_ = json.Unmarshal(gp, &gpr)
	if len(gpr.Matches) != 1 || gpr.Matches[0].Line != 2 {
		t.Fatalf("grep got %+v", gpr.Matches)
	}
}
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `search.go`** (uses `doublestar`-free `**` handling via manual walk; relative paths returned):

```go
package toolrunner

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type globArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"` // subdir under workdir; default "."
}
type globResult struct {
	Matches []string `json:"matches"`
	Error   string   `json:"error,omitempty"`
}

// matchGlob supports a leading "**/" meaning "any depth", plus filepath.Match
// semantics on the final segment. Good enough for the common cases.
func matchGlob(pattern, rel string) bool {
	if strings.HasPrefix(pattern, "**/") {
		tail := pattern[3:]
		base := filepath.Base(rel)
		if ok, _ := filepath.Match(tail, base); ok {
			return true
		}
		// also allow matching the full tail against the path
		ok, _ := filepath.Match(tail, rel)
		return ok
	}
	ok, _ := filepath.Match(pattern, rel)
	return ok
}

func (r *LocalToolRunner) walkFiles(sub string, fn func(rel string) error) error {
	base := r.cfg.WorkDir
	start := base
	if sub != "" {
		full, err := r.resolve(sub)
		if err != nil {
			return err
		}
		start = full
	}
	return filepath.WalkDir(start, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, rerr := filepath.Rel(base, p)
		if rerr != nil {
			return nil
		}
		return fn(filepath.ToSlash(rel))
	})
}

func (r *LocalToolRunner) runGlob(raw json.RawMessage) (json.RawMessage, error) {
	var a globArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("glob: bad args: %w", err)
	}
	var out []string
	err := r.walkFiles(a.Path, func(rel string) error {
		if matchGlob(a.Pattern, rel) {
			out = append(out, rel)
		}
		return nil
	})
	if err != nil {
		return result(globResult{Error: err.Error()})
	}
	return result(globResult{Matches: out})
}

type grepArgs struct {
	Pattern string `json:"pattern"`
	Glob    string `json:"glob,omitempty"`
	Path    string `json:"path,omitempty"`
}
type grepMatch struct {
	File string `json:"file"`
	Line int    `json:"line"`
	Text string `json:"text"`
}
type grepResult struct {
	Matches []grepMatch `json:"matches"`
	Error   string      `json:"error,omitempty"`
}

func (r *LocalToolRunner) runGrep(raw json.RawMessage) (json.RawMessage, error) {
	var a grepArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("grep: bad args: %w", err)
	}
	re, err := regexp.Compile(a.Pattern)
	if err != nil {
		return result(grepResult{Error: "bad regexp: " + err.Error()})
	}
	var out []grepMatch
	werr := r.walkFiles(a.Path, func(rel string) error {
		if a.Glob != "" && !matchGlob(a.Glob, rel) {
			return nil
		}
		f, err := os.Open(filepath.Join(r.cfg.WorkDir, rel))
		if err != nil {
			return nil
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		ln := 0
		for sc.Scan() {
			ln++
			if re.MatchString(sc.Text()) {
				out = append(out, grepMatch{File: rel, Line: ln, Text: sc.Text()})
			}
		}
		return nil
	})
	if werr != nil {
		return result(grepResult{Error: werr.Error()})
	}
	return result(grepResult{Matches: out})
}
```

- [ ] **Step 4: Run, expect PASS** — `go test ./toolrunner/ -run TestGlobAndGrep -v`.
- [ ] **Step 5: Commit** — `git commit -am "feat(toolrunner): glob + grep (in-process walk, .git-skipping)"`

---

## Task 4: `web_fetch` / `web_extract` lynxai client (TDD with httptest)

**Files:** Create `toolrunner/web.go`, `toolrunner/web_test.go`. Replace the two stubs.

> lynxai contract (from `reference_lynxai` / repo): `POST /fetch` body `{"url":...}` → `{"markdown":...}`; `POST /extract` body `{"url":...,"schema":{...}}` → structured JSON. Bearer key optional. Confirm field names against lynxai's `internal/api` request/response structs during the build; adjust the structs below to match.

- [ ] **Step 1: Failing test** — `web_test.go` (uses `httptest`, no real network):

```go
package toolrunner

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebFetchHitsLynxai(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/fetch" {
			t.Errorf("path %s", req.URL.Path)
		}
		body, _ := io.ReadAll(req.Body)
		if !strings.Contains(string(body), "example.com") {
			t.Errorf("body %s", body)
		}
		_, _ = w.Write([]byte(`{"markdown":"# Hello"}`))
	}))
	defer srv.Close()

	r, _ := New(Config{WorkDir: t.TempDir(), LynxaiBaseURL: srv.URL})
	out, err := r.runWebFetch(context.Background(), json.RawMessage(`{"url":"https://example.com"}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), "# Hello") {
		t.Fatalf("got %s", out)
	}
}

func TestWebFetchDisabledWhenNoEndpoint(t *testing.T) {
	r, _ := New(Config{WorkDir: t.TempDir()})
	out, err := r.runWebFetch(context.Background(), json.RawMessage(`{"url":"https://x"}`))
	if err != nil {
		t.Fatalf("should be tool result not Go error: %v", err)
	}
	if !strings.Contains(string(out), "error") {
		t.Fatalf("expected disabled error, got %s", out)
	}
}
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `web.go`**

```go
package toolrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type webFetchArgs struct {
	URL string `json:"url"`
}
type webExtractArgs struct {
	URL    string          `json:"url"`
	Schema json.RawMessage `json:"schema"`
}

// lynxaiPost posts body to {base}{path} and returns the raw response bytes.
// HTTP/transport failures become a tool-level error result (so the model can react).
func (r *LocalToolRunner) lynxaiPost(ctx context.Context, path string, body any) (json.RawMessage, error) {
	if r.cfg.LynxaiBaseURL == "" {
		return result(map[string]string{"error": "web access disabled: no lynxai endpoint configured"})
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("lynxai: marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.cfg.LynxaiBaseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("lynxai: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if r.cfg.LynxaiKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.cfg.LynxaiKey)
	}
	resp, err := r.cfg.HTTPClient.Do(req)
	if err != nil {
		return result(map[string]string{"error": "lynxai unreachable: " + err.Error()})
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, int64(r.cfg.MaxOutputBytes)))
	if resp.StatusCode != http.StatusOK {
		return result(map[string]string{"error": fmt.Sprintf("lynxai %s: status %d: %s", path, resp.StatusCode, string(raw))})
	}
	return json.RawMessage(raw), nil
}

func (r *LocalToolRunner) runWebFetch(ctx context.Context, raw json.RawMessage) (json.RawMessage, error) {
	var a webFetchArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("web_fetch: bad args: %w", err)
	}
	return r.lynxaiPost(ctx, "/fetch", webFetchArgs{URL: a.URL})
}

func (r *LocalToolRunner) runWebExtract(ctx context.Context, raw json.RawMessage) (json.RawMessage, error) {
	var a webExtractArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("web_extract: bad args: %w", err)
	}
	return r.lynxaiPost(ctx, "/extract", a)
}
```

- [ ] **Step 4: Run, expect PASS** — `go test ./toolrunner/ -run TestWeb -v`.
- [ ] **Step 5: Commit** — `git commit -am "feat(toolrunner): web_fetch/web_extract lynxai HTTP client"`

---

## Task 5: Tool schemas — `Defs()` (TDD)

**Files:** Create `toolrunner/defs.go`, `toolrunner/defs_test.go`.

- [ ] **Step 1: Failing test** — `defs_test.go`:

```go
package toolrunner

import (
	"encoding/json"
	"testing"
)

func TestDefsCoverAllTools(t *testing.T) {
	want := []string{"bash", "read", "write", "edit", "glob", "grep", "web_fetch", "web_extract"}
	defs := Defs()
	got := map[string]bool{}
	for _, d := range defs {
		got[d.Name] = true
		if len(d.InputSchema) == 0 {
			t.Errorf("%s: empty schema", d.Name)
		}
		var js map[string]any
		if err := json.Unmarshal(d.InputSchema, &js); err != nil {
			t.Errorf("%s: schema not valid JSON: %v", d.Name, err)
		}
	}
	for _, w := range want {
		if !got[w] {
			t.Errorf("missing tool def %q", w)
		}
	}
}
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `defs.go`** — one `ToolDef` per tool; schemas are JSON Schema objects matching each handler's args struct. Example for `bash` + `edit`; replicate the pattern for the rest (read/write/glob/grep/web_fetch/web_extract):

```go
package toolrunner

import (
	"encoding/json"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func obj(s string) json.RawMessage { return json.RawMessage(s) }

// Defs returns the JSON-schema tool surface for the local lane. The funnel
// passes these as TurnRequest.Tools alongside the LocalToolRunner.
func Defs() []bridle.ToolDef {
	return []bridle.ToolDef{
		{
			Name:        "bash",
			Description: "Run a shell command in the aspect's working directory. Returns stdout, stderr, exit_code.",
			InputSchema: obj(`{"type":"object","properties":{"command":{"type":"string","description":"the shell command"},"timeout_ms":{"type":"integer","description":"optional per-call timeout in ms"}},"required":["command"]}`),
		},
		{
			Name:        "read",
			Description: "Read a UTF-8 text file relative to the working directory.",
			InputSchema: obj(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
		},
		{
			Name:        "write",
			Description: "Create or overwrite a file relative to the working directory.",
			InputSchema: obj(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`),
		},
		{
			Name:        "edit",
			Description: "Replace old_string with new_string in a file. Fails if old_string is not unique unless replace_all is true.",
			InputSchema: obj(`{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]}`),
		},
		{
			Name:        "glob",
			Description: "Find files by glob pattern (supports a leading **/ for any depth). Returns paths relative to the working directory.",
			InputSchema: obj(`{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string","description":"optional subdirectory to search under"}},"required":["pattern"]}`),
		},
		{
			Name:        "grep",
			Description: "Search file contents by Go regular expression. Optionally restrict to files matching a glob.",
			InputSchema: obj(`{"type":"object","properties":{"pattern":{"type":"string"},"glob":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}`),
		},
		{
			Name:        "web_fetch",
			Description: "Fetch a web page as cleaned markdown (via the lynxai browser service).",
			InputSchema: obj(`{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}`),
		},
		{
			Name:        "web_extract",
			Description: "Extract structured JSON from a web page given a JSON schema (via lynxai).",
			InputSchema: obj(`{"type":"object","properties":{"url":{"type":"string"},"schema":{"type":"object"}},"required":["url","schema"]}`),
		},
	}
}
```

- [ ] **Step 4: Run, expect PASS** — `go test ./toolrunner/ -run TestDefs -v`.
- [ ] **Step 5: Full package test + vet** — `go test ./toolrunner/ && go vet ./toolrunner/`.
- [ ] **Step 6: Commit** — `git commit -am "feat(toolrunner): tool schemas (Defs)"`

---

## Task 6: Router + unknown-tool (TDD)

**Files:** `toolrunner/local_test.go` (router already implemented in Task 0).

- [ ] **Step 1: Test** — `local_test.go`:

```go
package toolrunner

import (
	"context"
	"encoding/json"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func TestRunRoutesAndRejectsUnknown(t *testing.T) {
	r := newTestRunner(t)
	if _, err := r.Run(context.Background(), bridle.ToolCall{Name: "bash", Args: json.RawMessage(`{"command":"true"}`)}); err != nil {
		t.Fatalf("bash route: %v", err)
	}
	if _, err := r.Run(context.Background(), bridle.ToolCall{Name: "frobnicate"}); err == nil {
		t.Fatal("unknown tool must return a Go error")
	}
}
```

- [ ] **Step 2: Run, expect PASS** (router exists) — `go test ./toolrunner/ -run TestRunRoutes -v`.
- [ ] **Step 3: Commit** — `git commit -am "test(toolrunner): router + unknown-tool"`

---

## Task 7: PROVEN CHAIN — live DeepSeek e2e (env-gated)

**Files:** Create `toolrunner/e2e_test.go`. This is the deliverable: a real model drives a multi-step tool chain.

> Gated by `BRIDLE_E2E_OPENAI_KEY` (+ optional `BRIDLE_E2E_OPENAI_BASE`, default DeepSeek). Run with the start-plumb DeepSeek key:
> `BRIDLE_E2E_OPENAI_KEY=$OPENAI_API_KEY BRIDLE_E2E_OPENAI_BASE=https://api.deepseek.com/v1 go test ./toolrunner/ -run TestE2E -v`
> (the user authorizes using the start-plumb.sh DeepSeek key for this; never hardcode it).

- [ ] **Step 1: Write the e2e test**

```go
package toolrunner

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	openai "github.com/CarriedWorldUniverse/bridle/provider/openai"
)

func TestE2EDeepSeekToolChain(t *testing.T) {
	key := os.Getenv("BRIDLE_E2E_OPENAI_KEY")
	if key == "" {
		t.Skip("set BRIDLE_E2E_OPENAI_KEY to run the live DeepSeek chain")
	}
	base := os.Getenv("BRIDLE_E2E_OPENAI_BASE")
	if base == "" {
		base = "https://api.deepseek.com/v1"
	}

	runner, err := New(Config{WorkDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	prov := openai.NewWithBaseURL(key, base)
	h := bridle.NewHarness(prov)

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	req := bridle.TurnRequest{
		AspectID: "e2e-test",
		Provider: bridle.ProviderOpenAI,
		Model:    "deepseek-chat",
		MaxSteps: 8,
		Tools:    Defs(),
		UserMessage: "Using the tools: 1) write a file notes.txt containing exactly the text BRIDLE_OK. " +
			"2) read it back. 3) run the bash command `wc -c < notes.txt`. " +
			"Then reply with the file's content and its byte count.",
	}

	res, err := h.RunTurn(ctx, req, runner, &sink{})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	t.Logf("stop=%s steps=%d tools=%d final=%q", res.StopReason, res.StepCount, len(res.ToolCalls), res.FinalText)

	// Proof the chain actually executed tools, not just hallucinated.
	var names []string
	for _, tc := range res.ToolCalls {
		names = append(names, tc.Name)
	}
	if len(res.ToolCalls) < 2 {
		t.Fatalf("expected a multi-step tool chain, got tools=%v", names)
	}
	if !strings.Contains(res.FinalText, "BRIDLE_OK") {
		t.Errorf("final text should report the file content; got %q", res.FinalText)
	}
}

type sink struct{}

func (sink) Emit(bridle.Event) {}
```

- [ ] **Step 2: Verify it SKIPS cleanly with no key** — `go test ./toolrunner/ -run TestE2E -v` → `SKIP`.

- [ ] **Step 3: Run it LIVE against DeepSeek**

```bash
cd ~/Source/bridle
BRIDLE_E2E_OPENAI_KEY="$(grep -o 'OPENAI_API_KEY:-[^}]*' ~/Source/start-plumb.sh | cut -d- -f2-)" \
BRIDLE_E2E_OPENAI_BASE=https://api.deepseek.com/v1 \
go test ./toolrunner/ -run TestE2E -v
```
Expected: PASS — log shows `tools=[write read bash ...]`, `final` contains `BRIDLE_OK` and a byte count (8).

- [ ] **Step 4: If it fails** — diagnose against the openai provider's tool-call lowering (does DeepSeek `/v1` return OpenAI `tool_calls`? does bridle's loop thread `tool` role results back?). Fix forward; the openai provider already has tool-replay tests (`run_session_tool_replay_test.go`) so the loop is known-good — most likely failure is a DeepSeek tool-schema quirk (adjust schemas) or a base-URL/model mismatch.

- [ ] **Step 5: Commit** — `git commit -am "test(toolrunner): live DeepSeek end-to-end tool chain (env-gated)"`

---

## Task 8: lynxai live web chain (optional, if lynxai is running)

- [ ] **Step 1:** Start lynxai locally (`cd ~/Source/lynxai && go run ./cmd/lynxai serve` with its DeepSeek env), note `:7878`.
- [ ] **Step 2:** Add `TestE2EWebFetchChain` (same shape as Task 7) with `Config{LynxaiBaseURL: "http://127.0.0.1:7878"}` and a prompt: "web_fetch https://github.com/CarriedWorldUniverse and tell me the org's tagline." Gate on `BRIDLE_E2E_LYNXAI` being set.
- [ ] **Step 3:** Run live; confirm `web_fetch` appears in `res.ToolCalls` and the answer reflects real page content.
- [ ] **Step 4:** Commit.

---

## Self-Review (against design §5, §12, §15-P1)

- **§5 local tool list:** bash ✓ read ✓ write ✓ edit ✓ glob ✓ grep ✓ web_fetch ✓ web_extract ✓. `web_search` is explicitly deferred (design §13 open knob) — noted, not built.
- **§12 error handling:** tool failures → tool_result (not Go error) ✓ (bash non-zero, file errors, lynxai unreachable, path escape); harness-level faults (bad args, unknown tool) → Go error ✓.
- **§12 testing:** per-lane unit tests ✓; lynxai client against fake server ✓; the design's "use `fake/tool_runner.go`" applies to *bridle's* loop tests — here we test the real runner directly + one live chain ✓.
- **§15 P1 definition** ("native-API turn loop end-to-end with a minimal tool set"): Task 7 is exactly that ✓.
- **No permission model / no host lane / no MCP:** correct — P2/P3/P4 ✓.
- **Type consistency:** result struct names (`bashResult`, `readResult`, `editResult`, `globResult`, `grepResult`, `grepMatch`) referenced in tests match definitions ✓. `New` returns `(*LocalToolRunner, error)` consistently ✓.
- **Open risk:** lynxai request/response field names (`markdown`, `schema`) are assumed from the reference note — Task 4 says verify against lynxai's `internal/api` structs during build.
