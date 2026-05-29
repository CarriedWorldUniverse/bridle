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
