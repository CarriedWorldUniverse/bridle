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

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/ctxmap/memory"
)

type attachment struct {
	eng   *memory.Engine
	tools map[string]memory.Tool

	// per-turn state, valid between BeforeModelCall(step 0) and OnTurnDone;
	// bridle runs one turn at a time per harness so no locking is needed
	turnN       int
	userMsg     string
	renderedIDs []string
}

// Attach registers the ctxmap hooks on h and returns a detach func.
func Attach(h *bridle.Harness, eng *memory.Engine) func() {
	a := &attachment{eng: eng, tools: map[string]memory.Tool{}}
	for _, t := range eng.Tools() {
		a.tools[t.Name] = t
	}
	ids := []bridle.HookID{
		h.RegisterBeforeModelCall(a.beforeModelCall),
		h.RegisterBeforeToolCall(a.beforeToolCall),
		h.RegisterOnTurnDone(a.onTurnDone),
	}
	return func() {
		for _, id := range ids {
			h.UnregisterHook(id)
		}
	}
}

func (a *attachment) beforeModelCall(_ context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
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

func (a *attachment) beforeToolCall(_ context.Context, in bridle.BeforeToolCallCtx) (bridle.BeforeToolCallCtx, bridle.HookAction, error) {
	t, ours := a.tools[in.Call.Name]
	if !ours {
		return in, bridle.HookContinue, nil
	}
	// serve from the engine via the deny pattern: harness skips execution
	// and uses Result as the tool_result
	out := t.Run(in.Call.Args)
	res, _ := json.Marshal(out)
	in.Deny = true
	in.Result = res
	return in, bridle.HookContinue, nil
}

func (a *attachment) onTurnDone(_ context.Context, in bridle.OnTurnDoneCtx) (bridle.OnTurnDoneCtx, bridle.HookAction, error) {
	a.eng.RecordTurn(a.turnN, a.userMsg, in.Result.FinalText, a.renderedIDs)
	a.renderedIDs = nil
	return in, bridle.HookContinue, nil
}
