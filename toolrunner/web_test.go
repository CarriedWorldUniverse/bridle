package toolrunner

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebFetchHitsLynxai(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/fetch" {
			t.Errorf("path %s", req.URL.Path)
		}
		body, _ := io.ReadAll(req.Body)
		if !strings.Contains(string(body), "example.com") {
			t.Errorf("body %s", body)
		}
		_, _ = w.Write([]byte(`{"markdown":"# Hello"}`))
	}))
	defer srv.Close()

	r, _ := New(Config{WorkDir: t.TempDir(), LynxaiBaseURL: srv.URL})
	out, err := r.runWebFetch(context.Background(), json.RawMessage(`{"url":"https://example.com"}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), "# Hello") {
		t.Fatalf("got %s", out)
	}
}

func TestWebFetchDisabledWhenNoEndpoint(t *testing.T) {
	r, _ := New(Config{WorkDir: t.TempDir()})
	out, err := r.runWebFetch(context.Background(), json.RawMessage(`{"url":"https://x"}`))
	if err != nil {
		t.Fatalf("should be tool result not Go error: %v", err)
	}
	if !strings.Contains(string(out), "error") {
		t.Fatalf("expected disabled error, got %s", out)
	}
}
