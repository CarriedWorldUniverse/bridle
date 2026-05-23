package bridle_test

import (
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/fake"
)

func TestAppendAssistantText(t *testing.T) {
	var finalText string
	var delta []bridle.SessionEvent

	bridle.AppendAssistantText(&finalText, &delta, bridle.ProviderClaude, "hello ")
	bridle.AppendAssistantText(&finalText, &delta, bridle.ProviderClaude, "world")

	if finalText != "hello world" {
		t.Errorf("finalText = %q; want %q", finalText, "hello world")
	}
	if len(delta) != 2 {
		t.Fatalf("delta len = %d; want 2", len(delta))
	}
	for i, want := range []string{"hello ", "world"} {
		ev := delta[i]
		if ev.Provider != bridle.ProviderClaude {
			t.Errorf("delta[%d].Provider = %q; want claude", i, ev.Provider)
		}
		if ev.Role != bridle.RoleAssistant {
			t.Errorf("delta[%d].Role = %q; want assistant", i, ev.Role)
		}
		if ev.Content != want {
			t.Errorf("delta[%d].Content = %q; want %q", i, ev.Content, want)
		}
	}
}

func TestEmitAssistantText(t *testing.T) {
	sink := &fake.SliceEventSink{}
	var finalText string
	var delta []bridle.SessionEvent

	bridle.EmitAssistantText(sink, &finalText, &delta, bridle.ProviderOpenAI, "hi")

	if finalText != "hi" {
		t.Errorf("finalText = %q; want hi", finalText)
	}
	if len(delta) != 1 || delta[0].Content != "hi" || delta[0].Provider != bridle.ProviderOpenAI {
		t.Errorf("delta = %+v; want one openai assistant 'hi'", delta)
	}
	if len(sink.Events) != 1 {
		t.Fatalf("sink events = %d; want 1", len(sink.Events))
	}
	mc, ok := sink.Events[0].(bridle.ModelChunk)
	if !ok {
		t.Fatalf("event type = %T; want ModelChunk", sink.Events[0])
	}
	if mc.Text != "hi" {
		t.Errorf("ModelChunk.Text = %q; want hi", mc.Text)
	}
}
