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
