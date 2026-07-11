// Package bridleadapter attaches a ctxmap memory engine to a bridle Harness
// using only bridle's existing hook seams — no bridle core changes:
//
//   - BeforeModelCall (step 0): assemble memory blocks — framing + epoch-frozen
//     core appended to the system prompt (stable prefix), the per-turn subgraph
//     inserted immediately before the final user message (churn at the prompt
//     END), and the recall/inspect tools added to the request.
//   - BeforeToolCall: serves recall/inspect from the engine via the
//     Deny+Result pattern (the harness skips execution and uses our result).
//   - OnTurnDone: feeds the completed turn back to the engine —
//     reuse-confirmation credit and async extraction.
//
// Wire it wherever the harness is constructed:
//
//	eng := memory.New(memory.Config{SessionID: repoID}, st, rend, prop, emb, judge)
//	detach := bridleadapter.Attach(h, eng)
//	defer detach() // and eng.Close()
//
// Attach is not safe to call concurrently with RunTurn (bridle's standing
// hook-registration rule). One adapter serves one Harness serving one
// conversation; sessions map 1:1 to engines.
package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/memory"
)

// dbg logs token-decomposition traces to stderr when CTXMAP_DEBUG is set, so a
// single run can be decomposed into memory-block overhead, memory-tool detours,
// and distiller savings without a rebuild.
var dbgOn = os.Getenv("CTXMAP_DEBUG") != ""

func dbg(format string, a ...interface{}) {
	if dbgOn {
		fmt.Fprintf(os.Stderr, "[ctxmap-dbg] "+format+"\n", a...)
	}
}

type attachment struct {
	eng   *memory.Engine
	tools map[string]memory.Tool

	// per-turn state, valid between BeforeModelCall(step 0) and OnTurnDone;
	// bridle runs one turn at a time per harness so no locking is needed
	turnN       int
	userMsg     string
	renderedIDs []string

	// within-turn mode (agentic coding): refresh the block every step +
	// ingest tool results as they stream
	withinTurn bool
	baseSys    string    // host AppendSystemPrompt before we add the memory section
	focus      string    // retrieval focus for mid-turn refreshes (the task)
	lastBlock  string    // last injected block; refresh the prompt only when it changes
	lastStepAt time.Time // per-step latency tracing (CTXMAP_DEBUG)
}

// Attach registers the ctxmap hooks on h and returns a detach func.
func Attach(h *bridle.Harness, eng *memory.Engine) func() {
	a := &attachment{eng: eng, tools: map[string]memory.Tool{}, withinTurn: eng.WithinTurnEnabled()}
	for _, t := range eng.Tools() {
		a.tools[t.Name] = t
	}
	ids := []bridle.HookID{
		h.RegisterBeforeModelCall(a.beforeModelCall),
		h.RegisterBeforeToolCall(a.beforeToolCall),
		h.RegisterAfterToolCall(a.afterToolCall),
		h.RegisterOnTurnDone(a.onTurnDone),
	}
	return func() {
		for _, id := range ids {
			h.UnregisterHook(id)
		}
	}
}

func (a *attachment) beforeModelCall(ctx context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
	if a.withinTurn {
		return a.beforeModelCallWithin(in)
	}
	if in.Step != 0 {
		return in, bridle.HookContinue, nil // in-loop steps append monotonically; nothing to add
	}
	a.turnN++

	// the final user message is the turn's input
	msgs := in.Request.Messages
	a.userMsg = ""
	lastUser := -1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			a.userMsg = msgs[i].Content
			lastUser = i
			break
		}
	}

	b := a.eng.AssembleBlocks(a.userMsg, a.turnN)
	a.renderedIDs = b.RenderedIDs
	dbg("turn=%d block: framing=%dch core=%dch subgraph=%dch (resent every step)",
		a.turnN, len(b.Framing), len(b.Core), len(b.Subgraph))

	// stable blocks join the system prompt
	sys := in.Request.AppendSystemPrompt
	if sys != "" {
		sys += "\n\n"
	}
	in.Request.AppendSystemPrompt = sys + b.Framing + "\n\n" + b.Core

	// churn goes at the END: subgraph as a user message just before the input
	if b.Subgraph != "" && lastUser >= 0 {
		injected := bridle.ProviderMessage{Role: "user", Content: b.Subgraph}
		msgs = append(msgs[:lastUser:lastUser], append([]bridle.ProviderMessage{injected}, msgs[lastUser:]...)...)
		in.Request.Messages = msgs
	}

	// memory tools
	for _, t := range a.eng.Tools() {
		in.Request.Tools = append(in.Request.Tools, bridle.ToolDef{
			Name: t.Name, Description: t.Description, InputSchema: t.InputSchema,
		})
	}
	return in, bridle.HookContinue, nil
}

// beforeModelCallWithin keeps the working-memory block in the SYSTEM PROMPT and
// REFRESHES it every step. The system prompt is never scrolled, so facts placed
// there don't degrade as the tool loop grows — the whole point of within-turn
// mode. baseSys (the host's own system prompt) is captured once and the memory
// section is rebuilt each step from the latest store state (which tool-result
// ingestion has been populating), so the block never accumulates or goes stale.
func (a *attachment) beforeModelCallWithin(in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
	if dbgOn {
		if !a.lastStepAt.IsZero() {
			dbg("step %d->%d took %.1fs", in.Step-1, in.Step, time.Since(a.lastStepAt).Seconds())
		}
		a.lastStepAt = time.Now()
	}
	if in.Step == 0 {
		a.turnN++
		msgs := in.Request.Messages
		a.userMsg = ""
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == "user" {
				a.userMsg = msgs[i].Content
				break
			}
		}
		a.focus = a.userMsg
		a.baseSys = in.Request.AppendSystemPrompt
		// maintain epoch consolidation + reuse-credit ids for RecordTurn
		b := a.eng.AssembleBlocks(a.userMsg, a.turnN)
		a.renderedIDs = b.RenderedIDs
		for _, t := range a.eng.Tools() {
			in.Request.Tools = append(in.Request.Tools, bridle.ToolDef{
				Name: t.Name, Description: t.Description, InputSchema: t.InputSchema,
			})
		}
	}
	block := a.eng.WorkingMemoryBlock(a.focus)
	// Refresh the system prompt ONLY when the block content changed. Rewriting it
	// every step would bust the prefix cache on every call (my own spec's cache-
	// stability invariant); facts land rarely, so most steps keep a stable prefix
	// and only a genuine new fact triggers one re-prefill.
	if in.Step == 0 || block != a.lastBlock {
		base := a.baseSys
		if base != "" {
			base += "\n\n"
		}
		in.Request.AppendSystemPrompt = base + memory.Framing + "\n\n" + block
		a.lastBlock = block
		// dump the CONTENT, not just the size — the experiment's quality question
		// is "are the injected facts the load-bearing ones?", answerable only by
		// reading what the model was actually given.
		dbg("within step=%d block=%dch (system-prompt REFRESHED); content:\n----8<----\n%s\n---->8----", in.Step, len(block), block)
	} else {
		dbg("within step=%d block=%dch (unchanged, prefix stable)", in.Step, len(block))
	}
	return in, bridle.HookContinue, nil
}

func (a *attachment) beforeToolCall(_ context.Context, in bridle.BeforeToolCallCtx) (bridle.BeforeToolCallCtx, bridle.HookAction, error) {
	t, ours := a.tools[in.Call.Name]
	if !ours {
		dbg("tool HOST  %s", in.Call.Name)
		return in, bridle.HookContinue, nil
	}
	dbg("tool MEM   %s  <- memory-tool detour (map-off never makes this call)", in.Call.Name)
	// serve from the engine via the deny pattern: harness skips execution
	// and uses Result as the tool_result
	out := t.Run(in.Call.Args)
	res, _ := json.Marshal(out)
	in.Deny = true
	in.Result = res
	return in, bridle.HookContinue, nil
}

func (a *attachment) afterToolCall(_ context.Context, in bridle.AfterToolCallCtx) (bridle.AfterToolCallCtx, bridle.HookAction, error) {
	if _, ours := a.tools[in.Call.Name]; ours || in.Call.Name == "read_raw" {
		return in, bridle.HookContinue, nil // memory tools + read_raw are already short/verbatim-by-design; never re-distill
	}
	if in.Result.Err != "" {
		return in, bridle.HookContinue, nil // leave errors verbatim
	}
	raw := string(in.Result.Result)
	// within-turn: mine the FULL raw result for durable facts (async) before any
	// distillation — the extractor should see the real content, not a summary.
	// Skip write_file: its result echoes what the model just authored (no new
	// knowledge) and re-mining it would just backlog the extractor.
	if a.withinTurn && in.Call.Name != "write_file" {
		a.eng.IngestToolResult(in.Call.Name, raw, a.focus)
	}
	shown := a.eng.DistillToolResult(in.Call.Name, raw)
	if shown != raw {
		dbg("distill %s  raw=%dch -> shown=%dch (saved %dch)", in.Call.Name, len(raw), len(shown), len(raw)-len(shown))
		b, _ := json.Marshal(shown)
		in.Result.Result = b
	} else if len(raw) > 1500 {
		dbg("distill %s  raw=%dch -> NOT distilled (passed through)", in.Call.Name, len(raw))
	}
	return in, bridle.HookContinue, nil
}

func (a *attachment) onTurnDone(_ context.Context, in bridle.OnTurnDoneCtx) (bridle.OnTurnDoneCtx, bridle.HookAction, error) {
	a.eng.RecordTurn(a.turnN, a.userMsg, in.Result.FinalText, a.renderedIDs)
	a.renderedIDs = nil
	return in, bridle.HookContinue, nil
}
