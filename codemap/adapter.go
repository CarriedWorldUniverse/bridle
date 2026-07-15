package codemap

import (
	"context"
	"encoding/json"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// Attach wires the codemap engine to a harness via existing hook seams (the
// ctxmap pattern; zero bridle core changes):
//
//   - BeforeModelCall (step 0): add the code_* tool defs to the request.
//   - BeforeToolCall: serve code_* tools from the engine (Deny+Result).
//   - AfterToolCall: write_file -> re-index that file; run_command -> mtime
//     sweep. Any DRIFT is appended to the mutating tool's own result, so the
//     model sees "what no longer matches" at the exact step it broke it.
//
// Returns a detach func.
func Attach(h *bridle.Harness, eng *Engine) func() {
	a := &cmAttachment{eng: eng}
	ids := []bridle.HookID{
		h.RegisterBeforeModelCall(a.beforeModelCall),
		h.RegisterBeforeToolCall(a.beforeToolCall),
		h.RegisterAfterToolCall(a.afterToolCall),
	}
	return func() {
		for _, id := range ids {
			h.UnregisterHook(id)
		}
	}
}

type cmAttachment struct {
	eng *Engine
}

func toolDefs() []bridle.ToolDef {
	obj := func(p string) json.RawMessage {
		return json.RawMessage(`{"type":"object","properties":` + p + `}`)
	}
	return []bridle.ToolDef{
		{Name: "code_outline", Description: "Symbols of one source file (kind, signature, line span) plus its imports — WITHOUT reading the file. Prefer this over read_file when you need structure, not bodies.",
			InputSchema: obj(`{"path":{"type":"string","description":"repo-relative file path"}},"required":["path"]`)},
		{Name: "code_symbol", Description: "Find a definition by name across the repo: file:line + full signature. Prefer this over re-reading a file to recall a signature.",
			InputSchema: obj(`{"name":{"type":"string"}},"required":["name"]`)},
		{Name: "code_refs", Description: "Where a name is referenced across the repo (file: line numbers) — the blast radius of changing it.",
			InputSchema: obj(`{"name":{"type":"string"}},"required":["name"]`)},
		{Name: "code_body", Description: "Exact source text of one named function/class (its precise line span only). Use when the signature isn't enough.",
			InputSchema: obj(`{"name":{"type":"string"}},"required":["name"]`)},
		{Name: "code_diag", Description: "Current structural inconsistencies: parse errors and imports naming symbols that don't exist.",
			InputSchema: obj(`{}`)},
	}
}

func (a *cmAttachment) beforeModelCall(_ context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
	if in.Step != 0 {
		return in, bridle.HookContinue, nil
	}
	in.Request.Tools = append(in.Request.Tools, toolDefs()...)
	return in, bridle.HookContinue, nil
}

func (a *cmAttachment) beforeToolCall(_ context.Context, in bridle.BeforeToolCallCtx) (bridle.BeforeToolCallCtx, bridle.HookAction, error) {
	var args struct {
		Path string `json:"path"`
		Name string `json:"name"`
	}
	json.Unmarshal(in.Call.Args, &args)
	var out string
	switch in.Call.Name {
	case "code_outline":
		out = a.eng.Outline(args.Path)
	case "code_symbol":
		out = a.eng.Symbol(args.Name)
	case "code_refs":
		out = a.eng.Refs(args.Name)
	case "code_body":
		out = a.eng.Body(args.Name)
	case "code_diag":
		out = a.eng.Diag()
	default:
		return in, bridle.HookContinue, nil
	}
	res, _ := json.Marshal(out)
	in.Deny = true
	in.Result = res
	return in, bridle.HookContinue, nil
}

func (a *cmAttachment) afterToolCall(_ context.Context, in bridle.AfterToolCallCtx) (bridle.AfterToolCallCtx, bridle.HookAction, error) {
	if in.Result.Err != "" {
		return in, bridle.HookContinue, nil
	}
	var drift []string
	switch in.Call.Name {
	case "write_file":
		var args struct {
			Path string `json:"path"`
		}
		json.Unmarshal(in.Call.Args, &args)
		if args.Path != "" {
			drift = a.eng.Reindex(args.Path)
		}
	case "run_command":
		drift = a.eng.SweepChanged()
	default:
		return in, bridle.HookContinue, nil
	}
	if len(drift) == 0 {
		return in, bridle.HookContinue, nil
	}
	// append the drift report to the mutating tool's own result — the model
	// learns what it broke at the step it broke it, not at the next test run
	var cur string
	if json.Unmarshal(in.Result.Result, &cur) != nil {
		cur = string(in.Result.Result)
	}
	cur += "\n\n[codemap drift — structure no longer matches]:"
	for _, d := range drift {
		cur += "\n  - " + d
	}
	b, _ := json.Marshal(cur)
	in.Result.Result = b
	return in, bridle.HookContinue, nil
}
