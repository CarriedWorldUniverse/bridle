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
