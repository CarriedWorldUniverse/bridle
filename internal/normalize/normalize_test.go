package normalize_test

import (
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
	"github.com/CarriedWorldUniverse/bridle/internal/normalize"
)

func TestProviderErrorClass(t *testing.T) {
	cases := []struct {
		kind bridle.ProviderErrorKind
		want bridle.ErrorClass
	}{
		{bridle.ProviderErrorAuthFailed, bridle.ErrorClassAuth},
		{bridle.ProviderErrorRateLimit, bridle.ErrorClassRateLimit},
		{bridle.ProviderErrorNetworkError, bridle.ErrorClassNetwork},
		{bridle.ProviderErrorTimeout, bridle.ErrorClassNetwork},
		{bridle.ProviderErrorTLSError, bridle.ErrorClassNetwork},
		{bridle.ProviderErrorServerError, bridle.ErrorClassProvider},
		{bridle.ProviderErrorCrash, bridle.ErrorClassProvider},
		{bridle.ProviderErrorSubprocessExit, bridle.ErrorClassProvider},
		{bridle.ProviderErrorConfig, bridle.ErrorClassProvider},
	}
	for _, c := range cases {
		got := normalize.ProviderErrorClass(c.kind)
		if got != c.want {
			t.Errorf("ProviderErrorClass(%q) = %q, want %q", c.kind, got, c.want)
		}
	}
}
