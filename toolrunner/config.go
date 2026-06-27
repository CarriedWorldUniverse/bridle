package toolrunner

import (
	"net/http"
	"time"
)

// Config configures a LocalToolRunner. Zero values are filled with
// sane defaults by New.
type Config struct {
	// WorkDir is the root for relative file paths and the cwd for bash.
	// Required; New returns an error if empty.
	WorkDir string

	// LynxaiBaseURL is the base URL of the shared lynxai service
	// (e.g. "http://127.0.0.1:7878"). Empty disables web_fetch/web_extract
	// (they return a tool-level error telling the model web access is off).
	LynxaiBaseURL string
	// LynxaiKey is an optional bearer token for lynxai (reverse-proxy auth).
	LynxaiKey string

	// HTTPClient is used for lynxai calls. Nil → a client with a 60s timeout.
	HTTPClient *http.Client

	// BashTimeout caps a single bash command. Zero → 120s.
	BashTimeout time.Duration

	// MaxOutputBytes caps captured stdout/stderr and file reads. Zero → 1<<20 (1 MiB).
	MaxOutputBytes int
}
