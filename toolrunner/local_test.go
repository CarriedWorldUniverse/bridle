package toolrunner

import (
	"context"
	"encoding/json"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func TestRunRoutesAndRejectsUnknown(t *testing.T) {
	r := newTestRunner(t)
	if _, err := r.Run(context.Background(), bridle.ToolCall{Name: "bash", Args: json.RawMessage(`{"command":"true"}`)}); err != nil {
		t.Fatalf("bash route: %v", err)
	}
	if _, err := r.Run(context.Background(), bridle.ToolCall{Name: "frobnicate"}); err == nil {
		t.Fatal("unknown tool must return a Go error")
	}
}
