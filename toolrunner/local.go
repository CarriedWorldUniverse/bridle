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

func (r *LocalToolRunner) runWebFetch(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	return nil, fmt.Errorf("todo")
}
func (r *LocalToolRunner) runWebExtract(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	return nil, fmt.Errorf("todo")
}
