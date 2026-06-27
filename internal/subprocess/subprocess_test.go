package subprocess

import (
	"context"
	"errors"
	"io"
	"os/exec"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func TestClassify(t *testing.T) {
	patterns := []Pattern{
		{Kind: "auth_failed", Patterns: []string{"not logged in", "authentication"}, Message: "auth msg"},
		{Kind: "rate_limit", Patterns: []string{"rate_limit", "rate limited"}, Message: "rate msg"},
		{Kind: "timeout", Patterns: []string{"timeout", "timed out"}, Message: "timeout msg"},
	}

	cases := []struct {
		name     string
		stderr   string
		wantKind bridle.ProviderErrorKind
		wantMsg  string
		wantOK   bool
	}{
		{"no match", "something else entirely", "", "", false},
		{"empty stderr", "", "", "", false},
		{"case-insensitive match", "ERROR: Not Logged In.", "auth_failed", "auth msg", true},
		{"first group wins over later", "authentication timed out", "auth_failed", "auth msg", true},
		{"later group", "request timed out", "timeout", "timeout msg", true},
		{"substring inside payload", `{"error":{"type":"rate_limit_error"}}`, "rate_limit", "rate msg", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			kind, msg, ok := Classify(tc.stderr, patterns)
			if kind != tc.wantKind || msg != tc.wantMsg || ok != tc.wantOK {
				t.Errorf("Classify(%q) = (%q, %q, %v), want (%q, %q, %v)",
					tc.stderr, kind, msg, ok, tc.wantKind, tc.wantMsg, tc.wantOK)
			}
		})
	}
}

func TestSharedPatterns_ActionableClasses(t *testing.T) {
	cases := []struct {
		name     string
		stderr   string
		wantKind bridle.ProviderErrorKind
	}{
		// AUTH — the headline class. A codex 401 (expired auth.json /
		// wrong endpoint) that prints no friendly "not logged in" string
		// must still classify as AUTH, not fall through to exit-1.
		{"bare 401", "Error: request failed with status 401", bridle.ProviderErrorAuthFailed},
		{"403", "HTTP 403 Forbidden", bridle.ProviderErrorAuthFailed},
		{"unauthorized", "server returned: Unauthorized", bridle.ProviderErrorAuthFailed},
		{"invalid api key", "invalid api key provided", bridle.ProviderErrorAuthFailed},
		{"expired token", "token expired, please re-authenticate", bridle.ProviderErrorAuthFailed},
		// RATE_LIMIT
		{"429", "got HTTP 429 from upstream", bridle.ProviderErrorRateLimit},
		{"quota", "you have exceeded your quota", bridle.ProviderErrorRateLimit},
		{"too many requests", "429 Too Many Requests", bridle.ProviderErrorRateLimit},
		// NETWORK
		{"connection refused", "dial tcp 1.2.3.4:443: connection refused", bridle.ProviderErrorNetworkError},
		{"dns", "lookup api.example.com: no such host", bridle.ProviderErrorNetworkError},
		{"unreachable", "connect: network is unreachable", bridle.ProviderErrorNetworkError},
		// CONFIG
		{"binary not found", "exec: \"codex\": executable file not found in $PATH", bridle.ProviderErrorConfig},
		{"missing config", "missing config file at ~/.codex/config.toml", bridle.ProviderErrorConfig},
		{"command not found", "/bin/sh: codex: command not found", bridle.ProviderErrorConfig},
		// SUBPROCESS_CRASH
		{"segfault", "Segmentation fault (core dumped)", bridle.ProviderErrorCrash},
		{"oom", "fatal: out of memory", bridle.ProviderErrorCrash},
		{"killed", "Killed", bridle.ProviderErrorCrash},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			kind, msg, ok := Classify(tc.stderr, SharedPatterns("testcli"))
			if !ok {
				t.Fatalf("Classify(%q) did not match; want kind %q", tc.stderr, tc.wantKind)
			}
			if kind != tc.wantKind {
				t.Errorf("Classify(%q) kind = %q, want %q", tc.stderr, kind, tc.wantKind)
			}
			if !strings.HasPrefix(msg, "testcli:") {
				t.Errorf("message %q not prefixed with provider id", msg)
			}
			if strings.Contains(strings.ToLower(msg), "exit status") {
				t.Errorf("message leaks raw exit code: %q", msg)
			}
		})
	}
}

func TestClassifyWithFallback_ProviderWinsThenShared(t *testing.T) {
	// Provider-specific pattern takes precedence (CLI-accurate wording).
	provider := []Pattern{
		{Kind: bridle.ProviderErrorAuthFailed, Patterns: []string{"not logged in"}, Message: "provider-worded auth"},
	}

	t.Run("provider pattern wins", func(t *testing.T) {
		kind, msg, ok := ClassifyWithFallback("Not logged in.", "codexcli", provider)
		if !ok || kind != bridle.ProviderErrorAuthFailed {
			t.Fatalf("kind=%q ok=%v, want auth_failed", kind, ok)
		}
		if msg != "provider-worded auth" {
			t.Errorf("msg = %q, want provider-worded", msg)
		}
	})

	t.Run("shared fallback catches bare 401", func(t *testing.T) {
		// codex 401 case: provider table has no "401" entry, shared does.
		kind, msg, ok := ClassifyWithFallback("Error: status 401 unauthorized", "codexcli", provider)
		if !ok || kind != bridle.ProviderErrorAuthFailed {
			t.Fatalf("kind=%q ok=%v, want auth_failed via shared fallback", kind, ok)
		}
		if !strings.HasPrefix(msg, "codexcli:") {
			t.Errorf("shared message not prefixed with codexcli: %q", msg)
		}
	})

	t.Run("no match anywhere", func(t *testing.T) {
		_, _, ok := ClassifyWithFallback("some unrelated noise", "codexcli", provider)
		if ok {
			t.Error("expected no match")
		}
	})
}

func TestIsResumeNotFound(t *testing.T) {
	cases := []struct {
		name   string
		stderr string
		want   bool
	}{
		{"codex no such thread", "Error: no such thread 'abc-123'", true},
		{"codex session not found", "session not found", true},
		{"claude no conversation", "No conversation found with session id xyz", true},
		{"gemini invalid resume", "invalid resume index: 99", true},
		{"corrupt", "session file is corrupt", true},
		{"does not exist", "checkpoint 5 does not exist", true},
		// Must NOT trip on transient / auth — those retry/surface as-is.
		{"auth not resume", "401 unauthorized", false},
		{"rate not resume", "429 too many requests", false},
		{"network not resume", "connection refused", false},
		{"empty", "", false},
		{"unrelated", "some other failure", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsResumeNotFound(tc.stderr); got != tc.want {
				t.Errorf("IsResumeNotFound(%q) = %v, want %v", tc.stderr, got, tc.want)
			}
		})
	}
}

func TestScanJSONLines(t *testing.T) {
	t.Run("trims, skips empties, dispatches in order", func(t *testing.T) {
		input := "  {\"a\":1}  \n\n\t\n{\"b\":2}\nnot json\n"
		var got []string
		err := ScanJSONLines(strings.NewReader(input), func(line []byte) {
			got = append(got, string(line))
		})
		if err != nil {
			t.Fatalf("ScanJSONLines: %v", err)
		}
		want := []string{`{"a":1}`, `{"b":2}`, "not json"}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("lines = %q, want %q", got, want)
		}
	})

	t.Run("handles lines beyond default bufio limit", func(t *testing.T) {
		big := `{"payload":"` + strings.Repeat("x", 200*1024) + `"}`
		var n, maxLen int
		err := ScanJSONLines(strings.NewReader(big+"\n"), func(line []byte) {
			n++
			if len(line) > maxLen {
				maxLen = len(line)
			}
		})
		if err != nil {
			t.Fatalf("ScanJSONLines: %v", err)
		}
		if n != 1 || maxLen != len(big) {
			t.Errorf("n=%d maxLen=%d, want 1 line of %d bytes", n, maxLen, len(big))
		}
	})

	t.Run("line over 1MB cap returns scanner error", func(t *testing.T) {
		huge := strings.Repeat("y", 2*1024*1024)
		err := ScanJSONLines(strings.NewReader(huge), func([]byte) {})
		if err == nil {
			t.Fatalf("expected bufio.ErrTooLong-style error, got nil")
		}
	})

	t.Run("read error propagates", func(t *testing.T) {
		wantErr := errors.New("boom")
		err := ScanJSONLines(io.MultiReader(strings.NewReader("{\"a\":1}\n"), &failingReader{err: wantErr}), func([]byte) {})
		if !errors.Is(err, wantErr) {
			t.Errorf("err = %v, want %v", err, wantErr)
		}
	})
}

type failingReader struct{ err error }

func (f *failingReader) Read([]byte) (int, error) { return 0, f.err }

func TestLastUserPrompt(t *testing.T) {
	cases := []struct {
		name string
		msgs []bridle.ProviderMessage
		want string
	}{
		{"nil messages", nil, ""},
		{"no user messages", []bridle.ProviderMessage{
			{Role: "system", Content: "sys"},
			{Role: "assistant", Content: "hi"},
		}, ""},
		{"single user", []bridle.ProviderMessage{
			{Role: "user", Content: "hello"},
		}, "hello"},
		{"most recent user wins", []bridle.ProviderMessage{
			{Role: "user", Content: "first"},
			{Role: "assistant", Content: "reply"},
			{Role: "user", Content: "second"},
			{Role: "assistant", Content: "trailing"},
		}, "second"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := LastUserPrompt(tc.msgs); got != tc.want {
				t.Errorf("LastUserPrompt = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestMergeEnv(t *testing.T) {
	base := []string{"PATH=/usr/bin", "HOME=/home/x", "FOO=old"}

	t.Run("empty overlay returns base unchanged", func(t *testing.T) {
		got := MergeEnv(base, nil)
		if !reflect.DeepEqual(got, base) {
			t.Errorf("MergeEnv(base, nil) = %v, want %v", got, base)
		}
	})

	t.Run("overlay replaces and appends", func(t *testing.T) {
		got := MergeEnv(base, map[string]string{"FOO": "new", "BAR": "added"})
		sorted := append([]string(nil), got...)
		sort.Strings(sorted)
		want := []string{"BAR=added", "FOO=new", "HOME=/home/x", "PATH=/usr/bin"}
		if !reflect.DeepEqual(sorted, want) {
			t.Errorf("MergeEnv = %v, want (any order) %v", got, want)
		}
		// Replacement must happen in place, not append a duplicate.
		if len(got) != 4 {
			t.Errorf("len = %d, want 4 (no duplicate FOO)", len(got))
		}
	})

	t.Run("inputs are left unmodified", func(t *testing.T) {
		baseCopy := append([]string(nil), base...)
		_ = MergeEnv(base, map[string]string{"FOO": "new"})
		if !reflect.DeepEqual(base, baseCopy) {
			t.Errorf("base mutated: %v", base)
		}
	})
}

// TestWatchCancelSignalsOnCancel: cancelling the context must terminate
// a long-running child via the graceful signal well inside the grace
// window (sleep exits on SIGTERM, so no SIGKILL escalation is needed).
func TestWatchCancelSignalsOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}

	procExited := make(chan struct{})
	watcherDone := make(chan struct{})
	go func() {
		defer close(watcherDone)
		WatchCancel(ctx, cmd, procExited, TermSignal())
	}()

	cancel()

	waitDone := make(chan error, 1)
	go func() { waitDone <- cmd.Wait() }()

	// Unix: the term signal lands immediately and sleep dies well inside
	// 3s. Windows: Process.Signal(os.Interrupt) is not deliverable, so
	// the process only dies at WatchCancel's 5s grace-period Kill — the
	// deadline must sit beyond the grace window to assert that fallback.
	deadline := 3 * time.Second
	if runtime.GOOS == "windows" {
		deadline = 8 * time.Second
	}
	select {
	case err := <-waitDone:
		if err == nil {
			t.Fatalf("expected sleep to be killed by signal, got clean exit")
		}
	case <-time.After(deadline):
		_ = cmd.Process.Kill()
		t.Fatalf("process not terminated within %v of cancel", deadline)
	}
	close(procExited)

	select {
	case <-watcherDone:
	case <-time.After(2 * time.Second):
		t.Fatalf("watcher did not return after procExited closed")
	}
}

// TestWatchCancelNaturalExit: when the process exits on its own and
// procExited is closed, the watcher must return without signaling.
func TestWatchCancelNaturalExit(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := exec.Command("true")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}

	procExited := make(chan struct{})
	watcherDone := make(chan struct{})
	go func() {
		defer close(watcherDone)
		WatchCancel(ctx, cmd, procExited, TermSignal())
	}()

	if err := cmd.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	close(procExited)

	select {
	case <-watcherDone:
	case <-time.After(2 * time.Second):
		t.Fatalf("watcher did not return after natural exit")
	}
}
