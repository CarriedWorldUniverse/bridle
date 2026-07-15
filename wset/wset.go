// Package wset is working-set retention: a context-eviction policy for long
// agentic sessions that works WITH the trained re-verify-via-tools tendency of
// modern coding models instead of against it.
//
// The finding behind it (agora ctxmap research, arms A–F): under context
// pressure, memory stores of every shape — extracted facts, working-state
// blocks, symbol servers, agent-authored decision journals — failed to rescue
// agentic coding, because models are trained to treat tool results as the only
// truth and re-read rather than trust any block. The intervention that worked
// was pure curation: never evict the newest read of a file, age out everything
// else. Three model families, keep=2 eviction: DeepSeek 2/6→6/6 passes, GLM
// 1/3→3/3, Sonnet recovered the full-window cost floor — at ~4–20× FEWER
// tokens than the sliding-window baseline (re-reading was always paid in
// tokens too).
//
// Policy, per model call:
//
//   - The LATEST result of each read-class tool call, keyed by its target
//     (e.g. read_file's path), is never evicted. Older results for the SAME
//     key are stubbed as superseded.
//   - All other tool results keep the most recent KeepOthers verbatim; older
//     ones are stubbed.
//
// Context therefore scales with the size of the working set (files touched),
// not the length of the history. The policy only mutates a message when its
// content is superseded or ages past the window — retained content stays
// byte-identical at its original position, so provider prefix caching keeps
// paying for the retained bulk (a sliding window churns the prefix every step
// and forfeits the cache).
//
// Deterministic, zero model calls, hooks-only (the ctxmap/codemap attachment
// pattern; no bridle-core changes). Message STRUCTURE is never altered —
// deleting messages breaks assistant-toolcall/tool-result pairing on strict
// providers — only result content is replaced with a short stub.
package wset

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// Config tunes the retention policy. The zero value is NOT useful — use
// DefaultConfig() and override.
type Config struct {
	// ReadTools maps a tool name to the name of the string argument that
	// keys its working-set identity (read_file → "path"). The latest result
	// per (tool, key) is retained; earlier ones are superseded.
	ReadTools map[string]string

	// KeepOthers is how many non-read tool results are kept verbatim
	// (newest first). Older ones are stubbed. <=0 means keep none.
	KeepOthers int

	// MaxRetainBytes, when >0, truncates a RETAINED result that exceeds it
	// (head-kept, marker appended). Guards against a single giant command
	// output or file dominating the window. 0 = no cap.
	MaxRetainBytes int
}

// DefaultConfig retains the newest read_file per path, keeps the last 2 other
// tool results, and caps any retained result at 64 KiB.
func DefaultConfig() Config {
	return Config{
		ReadTools:      map[string]string{"read_file": "path", "Read": "file_path"},
		KeepOthers:     2,
		MaxRetainBytes: 64 << 10,
	}
}

// Attach registers the policy on the harness and returns a detach func.
func Attach(h *bridle.Harness, cfg Config) func() {
	a := &attachment{cfg: cfg}
	id := h.RegisterBeforeModelCall(a.beforeModelCall)
	return func() { h.UnregisterHook(id) }
}

const (
	stubEvicted    = "[tool result evicted: aged out of context window; %d chars dropped]"
	stubSuperseded = "[tool result superseded: a newer %s of %q appears later in context; %d chars dropped]"
	truncMarker    = "\n[…result truncated by working-set retention: %d of %d chars kept]"
)

// stubbed reports whether content is already one of our stubs (idempotence:
// the hook runs every step over the same history).
func stubbed(content string) bool {
	return strings.HasPrefix(content, "[tool result evicted:") ||
		strings.HasPrefix(content, "[tool result superseded:")
}

// truncateRetained caps a retained result at max bytes, idempotently: already-
// truncated content (marker present) is left byte-identical so the history
// stays prefix-stable across steps.
func truncateRetained(content string, max int) string {
	if max <= 0 || len(content) <= max || strings.Contains(content, "…result truncated") {
		return content
	}
	return content[:max] + fmt.Sprintf(truncMarker, max, len(content))
}

type attachment struct {
	cfg Config
}

func (a *attachment) beforeModelCall(_ context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
	msgs := in.Request.Messages

	// tool-call id -> (tool name, working-set key) from assistant tool_use
	// blocks. Only read-class calls with a non-empty key participate in
	// per-key retention; everything else is "other".
	type meta struct{ name, key string }
	calls := map[string]meta{}
	for _, m := range msgs {
		for _, tc := range m.ToolCalls {
			mt := meta{name: tc.Name}
			if argName, ok := a.cfg.ReadTools[tc.Name]; ok {
				var args map[string]json.RawMessage
				if json.Unmarshal(tc.Args, &args) == nil {
					var v string
					json.Unmarshal(args[argName], &v)
					mt.key = v
				}
			}
			calls[tc.ID] = mt
		}
	}

	// newest live result per (tool, key), found in one reverse pass
	newest := map[string]int{} // "tool\x00key" -> msg index
	var others []int           // live non-read results, oldest first
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m.Role != "tool_result" || stubbed(m.Content) {
			continue
		}
		c := calls[m.ToolCallID]
		if c.key != "" {
			k := c.name + "\x00" + c.key
			if _, seen := newest[k]; !seen {
				newest[k] = i
			}
		}
	}
	for i, m := range msgs {
		if m.Role != "tool_result" || stubbed(m.Content) {
			continue
		}
		c := calls[m.ToolCallID]
		if c.key != "" {
			if newest[c.name+"\x00"+c.key] != i {
				msgs[i].Content = fmt.Sprintf(stubSuperseded, c.name, c.key, len(m.Content))
			} else {
				msgs[i].Content = truncateRetained(m.Content, a.cfg.MaxRetainBytes)
			}
			continue
		}
		others = append(others, i)
	}
	for n := 0; n < len(others)-a.cfg.KeepOthers; n++ {
		i := others[n]
		msgs[i].Content = fmt.Sprintf(stubEvicted, len(msgs[i].Content))
	}
	start := len(others) - a.cfg.KeepOthers
	if start < 0 {
		start = 0
	}
	for _, i := range others[start:] {
		msgs[i].Content = truncateRetained(msgs[i].Content, a.cfg.MaxRetainBytes)
	}
	return in, bridle.HookContinue, nil
}
