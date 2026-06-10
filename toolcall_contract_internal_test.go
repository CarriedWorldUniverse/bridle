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
		{"im_start", "<|im_start|>system you are helpful<|im_end|>"},
		{"channel-commentary", "<|channel|>commentary scratch work here"},
	}
	for _, tc := range leaked {
		t.Run("leaked/"+tc.name, func(t *testing.T) {
			rep := detectLeak(ProviderResult{FinalText: tc.text})
			if !rep.detected {
				t.Errorf("detectLeak(%q) = not detected; want detected", tc.text)
			}
		})
	}
}

func TestDetectLeak_JSONToolCallAsText(t *testing.T) {
	cases := []string{
		`{"name": "send_chat", "arguments": {"text": "hi"}}`,
		`Sure: {"name":"deploy","args":{"env":"prod"}}`,
		`{"name": "noop", "parameters": null}`,
	}
	for _, c := range cases {
		rep := detectLeak(ProviderResult{FinalText: c})
		if !rep.detected {
			t.Errorf("detectLeak(%q) = not detected; want tool-call-as-text", c)
		}
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
	}
	for _, c := range clean {
		t.Run(c.name, func(t *testing.T) {
			if rep := detectLeak(c.res); rep.detected {
				t.Errorf("detectLeak flagged clean output as leak (detail=%q)", rep.detail)
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
	if rep := detectLeak(res); !rep.detected {
		t.Errorf("detectLeak missed leaked token inside tool-call args")
	}
}

// --- Repair ---

func TestRepairLeak_StripsTokensAroundCleanText(t *testing.T) {
	res := ProviderResult{FinalText: "<|channel|>final<|message|>The deploy is green.<|end|>"}
	out := repairLeak(res)
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
	out := repairLeak(res)
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
	out := repairLeak(res)
	if out.clean {
		t.Errorf("expected unrepairable garble to be not-clean (triggers retry); got clean=%+v", out)
	}
}

func TestRepairLeak_MalformedJSONToolCallNotExtracted(t *testing.T) {
	// name present but args aren't valid JSON — must not ship a broken call;
	// falls through to not-clean so strict mode retries.
	res := ProviderResult{FinalText: `{"name":"x","arguments":{not valid}}`}
	out := repairLeak(res)
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
