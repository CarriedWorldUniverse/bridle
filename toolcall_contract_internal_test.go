package bridle

import (
	"encoding/json"
	"testing"
)

// --- Detect: leak corpus vs clean controls ---

func TestDetectLeak_ProtocolTokenCorpus(t *testing.T) {
	// Real-shaped leak samples from the gemma A/B (ollama openai-compat shim
	// leaking Gemma's harmony/channel grammar) plus the JSON-as-text class.
	leaked := []struct {
		name string
		text string
	}{
		{"channel-thought", "<|channel|>thought\nLet me think about this.<|message|>The answer is 4."},
		{"tool_call-token", "I'll call the tool now <|tool_call|>{\"name\":\"x\"}"},
		{"malformed-tool_call", "result <tool_call|> here"},
		{"start-end-markers", "<|start|>assistant<|message|>hi<|end|>"},
		{"assistantfinal", "assistantfinal The deploy is green."},
		{"channel-commentary", "<|channel|>commentary scratch work here"},
		{"message-token", "<|message|>here is the body"},
		{"constrain-token", "<|constrain|>json"},
	}
	for _, tc := range leaked {
		t.Run("leaked/"+tc.name, func(t *testing.T) {
			rep := detectLeak(ProviderResult{FinalText: tc.text}, false)
			if !rep.detected {
				t.Errorf("detectLeak(%q) = not detected; want detected", tc.text)
			}
		})
	}
}

func TestDetectLeak_JSONToolCallAsText(t *testing.T) {
	// With tools defined, a JSON-as-text tool call IS a leak.
	cases := []string{
		`{"name": "send_chat", "arguments": {"text": "hi"}}`,
		`Sure: {"name":"deploy","args":{"env":"prod"}}`,
		`{"name": "noop", "parameters": null}`,
	}
	for _, c := range cases {
		rep := detectLeak(ProviderResult{FinalText: c}, true)
		if !rep.detected {
			t.Errorf("detectLeak(%q) = not detected; want tool-call-as-text", c)
		}
	}
}

// TestDetectLeak_JSONToolCallAsText_NoToolsNotDetected is the CRITICAL 2
// guard: a model with NO tools defined cannot have leaked a tool call, so a
// {"name":...,"arguments":...} blob in its prose is a documented example
// (JSON-RPC/MCP docs) and must NOT be flagged or extracted.
func TestDetectLeak_JSONToolCallAsText_NoToolsNotDetected(t *testing.T) {
	cases := []string{
		`{"name": "send_chat", "arguments": {"text": "hi"}}`,
		`Here's an MCP example: {"name":"search","arguments":{"query":"foo"}}`,
		`{"name": "noop", "parameters": null}`,
	}
	for _, c := range cases {
		rep := detectLeak(ProviderResult{FinalText: c}, false)
		if rep.detected {
			t.Errorf("detectLeak(%q, hasTools=false) flagged a documented example as a leak (detail=%q)", c, rep.detail)
		}
	}
}

// TestRepairLeak_NoToolsDoesNotExtract asserts repair preserves the prose: a
// no-tools turn whose text is a JSON-RPC/MCP example is returned untouched —
// no extraction, no spurious ToolInvocation.
func TestRepairLeak_NoToolsDoesNotExtract(t *testing.T) {
	in := `Here is an MCP example: {"name":"search","arguments":{"query":"foo"}}`
	out := repairLeak(ProviderResult{FinalText: in}, false)
	if out.extractedCall {
		t.Errorf("repairLeak extracted a tool call from a no-tools documented example")
	}
	if len(out.toolCalls) != 0 {
		t.Errorf("repairLeak synthesized %d tool call(s) from a no-tools example; want 0", len(out.toolCalls))
	}
	if out.text != in {
		t.Errorf("repairLeak mutated no-tools prose:\n  in:  %q\n  out: %q", in, out.text)
	}
}

func TestDetectLeak_CleanControlsPass(t *testing.T) {
	// These must NOT trip detection — false-positive leak detection is the
	// key risk (it would mangle valid output).
	clean := []struct {
		name string
		res  ProviderResult
	}{
		{"plain-prose", ProviderResult{FinalText: "The deployment is complete and green."}},
		{"final-in-prose", ProviderResult{FinalText: "This is my final answer: 42."}},
		{"commentary-in-prose", ProviderResult{FinalText: "I added a commentary block to the PR."}},
		{"json-data-no-name-first", ProviderResult{FinalText: `Here is data: {"status": "ok", "count": 3}`}},
		{"structured-call-clean", ProviderResult{
			FinalText: "Calling the tool.",
			ToolCalls: []ToolInvocation{{ID: "1", Name: "echo", Args: json.RawMessage(`{"msg":"hi"}`)}},
		}},
		{"json-name-but-structured-call-exists", ProviderResult{
			// The model explained a call in prose AND made a real one — the
			// prose JSON must not be mistaken for the intended invocation.
			FinalText: `I will run {"name":"echo","arguments":{}}`,
			ToolCalls: []ToolInvocation{{ID: "1", Name: "echo", Args: json.RawMessage(`{}`)}},
		}},
		{"angle-bracket-html", ProviderResult{FinalText: "Use the <div> tag and <span> element."}},
		// CRITICAL 1 clean controls: ChatML / Llama / role tokens are valid
		// prose when an aspect explains a prompt format or writes tokenizer
		// code. They are NOT harmony leaks and must pass clean + untouched.
		{"chatml-format-explained", ProviderResult{
			FinalText: "The ChatML format uses <|im_start|>system ... <|im_end|> as delimiters.",
		}},
		{"llama-format-explained", ProviderResult{
			FinalText: "Llama 3 wraps turns as <|begin_of_text|><|start_header_id|>user<|end_header_id|> ...",
		}},
		{"role-tokens-in-prose", ProviderResult{
			FinalText: "The role tokens are <|user|>, <|assistant|> and <|system|>.",
		}},
	}
	for _, c := range clean {
		t.Run(c.name, func(t *testing.T) {
			if rep := detectLeak(c.res, true); rep.detected {
				t.Errorf("detectLeak flagged clean output as leak (detail=%q)", rep.detail)
			}
		})
	}
}

// TestRepairLeak_CleanChatMLByteIdentical asserts that a turn whose text
// contains ChatML/Llama/role tokens in valid prose is NOT just undetected but
// also passes through repair byte-identical — no token gets silently stripped.
func TestRepairLeak_CleanChatMLByteIdentical(t *testing.T) {
	cases := []string{
		"The ChatML format uses <|im_start|>system ... <|im_end|> as delimiters.",
		"Llama 3 wraps turns as <|begin_of_text|><|start_header_id|>user<|end_header_id|> ...",
		"The role tokens are <|user|>, <|assistant|> and <|system|>.",
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			if stripped := stripLeakTokens(in); stripped != in {
				t.Errorf("stripLeakTokens mutated valid prose:\n  in:  %q\n  out: %q", in, stripped)
			}
		})
	}
}

func TestDetectLeak_LeakedTokensInArgs(t *testing.T) {
	res := ProviderResult{
		ToolCalls: []ToolInvocation{{
			ID: "1", Name: "echo",
			Args: json.RawMessage(`{"msg":"hi<|channel|>thought"}`),
		}},
	}
	if rep := detectLeak(res, true); !rep.detected {
		t.Errorf("detectLeak missed leaked token inside tool-call args")
	}
}

// --- Repair ---

func TestRepairLeak_StripsTokensAroundCleanText(t *testing.T) {
	res := ProviderResult{FinalText: "<|channel|>final<|message|>The deploy is green.<|end|>"}
	out := repairLeak(res, false)
	if !out.clean {
		t.Fatalf("repair not clean: %+v", out)
	}
	if out.text != "The deploy is green." {
		t.Errorf("repaired text = %q; want %q", out.text, "The deploy is green.")
	}
	if textCarriesLeak(out.text) {
		t.Errorf("repaired text still carries a leak token: %q", out.text)
	}
}

func TestRepairLeak_ExtractsToolCallFromText(t *testing.T) {
	res := ProviderResult{
		FinalText: `Sure, I'll do that. {"name":"send_chat","arguments":{"text":"hello"}}`,
	}
	out := repairLeak(res, true)
	if !out.clean {
		t.Fatalf("repair not clean: %+v", out)
	}
	if !out.extractedCall {
		t.Fatalf("expected a tool call extracted from text")
	}
	if len(out.toolCalls) != 1 {
		t.Fatalf("toolCalls len = %d; want 1", len(out.toolCalls))
	}
	got := out.toolCalls[0]
	if got.Name != "send_chat" {
		t.Errorf("extracted call name = %q; want send_chat", got.Name)
	}
	var args struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(got.Args, &args); err != nil {
		t.Fatalf("extracted args not valid JSON: %v", err)
	}
	if args.Text != "hello" {
		t.Errorf("extracted args.text = %q; want hello", args.Text)
	}
	// The JSON blob must be removed from the surfaced text.
	if textCarriesLeak(out.text) {
		t.Errorf("residual leak in text after extraction: %q", out.text)
	}
}

func TestRepairLeak_UnrepairableGarbleNotClean(t *testing.T) {
	// Only protocol tokens, no recoverable content: stripping leaves nothing.
	res := ProviderResult{FinalText: "<|channel|><|message|><|end|>"}
	out := repairLeak(res, false)
	if out.clean {
		t.Errorf("expected unrepairable garble to be not-clean (triggers retry); got clean=%+v", out)
	}
}

// TestBuildInvocationFromText_UniqueIDsPerExtraction is the IMPORTANT 4
// guard: two extractions in one turn (different call content) must produce
// DISTINCT ids, or strict providers reject the message history.
func TestBuildInvocationFromText_UniqueIDsPerExtraction(t *testing.T) {
	a, okA := buildInvocationFromText("search", `{"query":"foo"}`)
	b, okB := buildInvocationFromText("deploy", `{"env":"prod"}`)
	if !okA || !okB {
		t.Fatalf("expected both extractions to succeed: okA=%v okB=%v", okA, okB)
	}
	if a.ID == b.ID {
		t.Errorf("two extractions produced duplicate id %q; want distinct", a.ID)
	}
	if a.ID == "" || b.ID == "" {
		t.Errorf("synthesized id must be non-empty: a=%q b=%q", a.ID, b.ID)
	}
	// Determinism: identical content yields the identical id (so a re-run /
	// re-extraction of the same call is stable).
	a2, _ := buildInvocationFromText("search", `{"query":"foo"}`)
	if a.ID != a2.ID {
		t.Errorf("identical content gave different ids: %q vs %q", a.ID, a2.ID)
	}
}

func TestRepairLeak_MalformedJSONToolCallNotExtracted(t *testing.T) {
	// name present but args aren't valid JSON — must not ship a broken call;
	// falls through to not-clean so strict mode retries.
	res := ProviderResult{FinalText: `{"name":"x","arguments":{not valid}}`}
	out := repairLeak(res, true)
	if out.extractedCall {
		t.Errorf("must not extract a call from malformed JSON args")
	}
}

func TestStripToolCallArgs_CleanArgsByteIdentical(t *testing.T) {
	// A clean call's args must pass through byte-identical — the contract
	// must never mangle valid tool calls.
	orig := []ToolInvocation{{ID: "1", Name: "echo", Args: json.RawMessage(`{"msg":"hi"}`)}}
	out := stripToolCallArgs(orig)
	if string(out[0].Args) != `{"msg":"hi"}` {
		t.Errorf("clean args mangled: %q", string(out[0].Args))
	}
}

func TestEffectiveStrictness_Default(t *testing.T) {
	if got := effectiveStrictness(""); got != ToolCallStrictnessRepairThenRetry {
		t.Errorf("default strictness = %q; want repair-then-retry", got)
	}
	if got := effectiveStrictness(ToolCallStrictnessTolerant); got != ToolCallStrictnessTolerant {
		t.Errorf("strictness = %q; want tolerant", got)
	}
}
