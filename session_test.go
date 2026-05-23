package bridle_test

import (
	"encoding/json"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// TestParseSessionEvent_PerProviderRawJSON pins the per-provider
// RawJSON shapes that ParseSessionEvent's switch dispatches on. If a
// provider implementation stops setting Provider or changes the
// RawJSON shape, ParseSessionEvent silently falls through to the
// "unknown" branch and returns the raw bytes as content — operators
// looking at session history then see {"type":"tool_use",...} blobs
// instead of "tool_use: foo {...}" summaries. These tests guard the
// contract from the parsing side.
func TestParseSessionEvent_PerProviderRawJSON(t *testing.T) {
	cases := []struct {
		name     string
		provider bridle.ProviderID
		raw      string
		want     string
	}{
		{
			name:     "claude tool_use block",
			provider: bridle.ProviderClaude,
			raw:      `{"type":"tool_use","id":"abc","name":"send_chat","input":{"content":"hi"}}`,
			want:     `tool_use: send_chat {"content":"hi"}`,
		},
		{
			name:     "claudecode tool_use block",
			provider: bridle.ProviderClaudeCode,
			raw:      `{"type":"tool_use","id":"abc","name":"Bash","input":{"command":"ls"}}`,
			want:     `tool_use: Bash {"command":"ls"}`,
		},
		{
			name:     "openai tool_call",
			provider: bridle.ProviderOpenAI,
			raw:      `{"id":"call_1","function":{"name":"send_chat","arguments":"{\"content\":\"hi\"}"}}`,
			want:     `tool_use: send_chat {"content":"hi"}`,
		},
		{
			name:     "ollama tool_call",
			provider: bridle.ProviderOllama,
			raw:      `{"function":{"name":"send_chat","arguments":{"content":"hi"}}}`,
			want:     `tool_use: send_chat {"content":"hi"}`,
		},
		{
			name:     "bedrock tool_use",
			provider: bridle.ProviderBedrock,
			raw:      `{"name":"send_chat","input":{"content":"hi"}}`,
			want:     `tool_use: send_chat {"content":"hi"}`,
		},
		{
			name:     "gemini functionCall",
			provider: bridle.ProviderGemini,
			raw:      `{"functionCall":{"name":"send_chat","args":{"content":"hi"}}}`,
			want:     `tool_use: send_chat {"content":"hi"}`,
		},
		{
			name:     "geminicli tool_use",
			provider: bridle.ProviderGeminiCLI,
			raw:      `{"type":"tool_use","tool_name":"send_chat","parameters":{"content":"hi"}}`,
			want:     `tool_use: send_chat {"content":"hi"}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ev := bridle.SessionEvent{
				Provider: tc.provider,
				Role:     bridle.RoleAssistant,
				RawJSON:  json.RawMessage(tc.raw),
			}
			got, err := bridle.ParseSessionEvent(ev)
			if err != nil {
				t.Fatalf("ParseSessionEvent: %v", err)
			}
			if got.Content != tc.want {
				t.Errorf("Content = %q; want %q", got.Content, tc.want)
			}
		})
	}
}

// TestParseSessionEvent_UnknownProviderFallsThrough confirms the
// "unknown provider" branch in ParseSessionEvent returns the raw bytes
// as content. This is the behaviour that silently kicks in whenever a
// provider implementation forgets to set SessionEvent.Provider — the
// reason every provider must set it.
func TestParseSessionEvent_UnknownProviderFallsThrough(t *testing.T) {
	raw := `{"type":"tool_use","name":"foo","input":{}}`
	ev := bridle.SessionEvent{
		// Provider intentionally unset — simulating the bug class.
		Role:    bridle.RoleAssistant,
		RawJSON: json.RawMessage(raw),
	}
	got, err := bridle.ParseSessionEvent(ev)
	if err != nil {
		t.Fatalf("ParseSessionEvent: %v", err)
	}
	if !strings.Contains(got.Content, "tool_use") || !strings.Contains(got.Content, "{") {
		t.Errorf("expected raw-bytes fallback; got %q", got.Content)
	}
}
