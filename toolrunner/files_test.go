package toolrunner

import (
	"encoding/json"
	"testing"
)

func TestWriteThenRead(t *testing.T) {
	r := newTestRunner(t)
	if _, err := r.runWrite(json.RawMessage(`{"path":"a.txt","content":"hi there"}`)); err != nil {
		t.Fatal(err)
	}
	out, err := r.runRead(json.RawMessage(`{"path":"a.txt"}`))
	if err != nil {
		t.Fatal(err)
	}
	var res readResult
	_ = json.Unmarshal(out, &res)
	if res.Content != "hi there" {
		t.Fatalf("got %q", res.Content)
	}
}

func TestEditReplace(t *testing.T) {
	r := newTestRunner(t)
	_, _ = r.runWrite(json.RawMessage(`{"path":"b.txt","content":"foo bar foo"}`))
	out, err := r.runEdit(json.RawMessage(`{"path":"b.txt","old_string":"foo","new_string":"baz","replace_all":true}`))
	if err != nil {
		t.Fatal(err)
	}
	var res editResult
	_ = json.Unmarshal(out, &res)
	if res.Replacements != 2 {
		t.Fatalf("want 2 replacements, got %+v", res)
	}
	rd, _ := r.runRead(json.RawMessage(`{"path":"b.txt"}`))
	var got readResult
	_ = json.Unmarshal(rd, &got)
	if got.Content != "baz bar baz" {
		t.Fatalf("got %q", got.Content)
	}
}

func TestPathEscapeRejected(t *testing.T) {
	r := newTestRunner(t)
	out, err := r.runRead(json.RawMessage(`{"path":"../../etc/passwd"}`))
	if err != nil {
		t.Fatalf("escape should be a tool result, not a Go error: %v", err)
	}
	var res readResult
	_ = json.Unmarshal(out, &res)
	if res.Error == "" {
		t.Fatalf("expected an error field for escaping path, got %+v", res)
	}
}
