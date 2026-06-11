package openai

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/openai/openai-go"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// deepSeekHost is the host of DeepSeek's OpenAI-compatible API. The
// provider is "DeepSeek-aware" when its configured baseURL points here —
// detection is host-based so it survives scheme / path / trailing-slash
// variation.
const deepSeekHost = "api.deepseek.com"

// isDeepSeekEndpoint reports whether baseURL targets DeepSeek's
// OpenAI-compatible /v1. Empty baseURL = SDK default (api.openai.com) =
// not DeepSeek. Host comparison is case-insensitive.
//
// NEX-587: DeepSeek is reached through the openai provider pointed at
// api.deepseek.com (OpenAI-compatible wire), so the provider can't tell
// it's talking to DeepSeek from the model id alone — the baseURL is the
// signal. Per-endpoint capability gating (strict json_schema
// unsupported, reasoning_content extension) keys off this.
func isDeepSeekEndpoint(baseURL string) bool {
	if baseURL == "" {
		return false
	}
	u, err := url.Parse(baseURL)
	if err != nil {
		// Fall back to a substring check for un-parseable inputs rather
		// than silently returning false on a malformed-but-DeepSeek URL.
		return strings.Contains(strings.ToLower(baseURL), deepSeekHost)
	}
	host := u.Hostname()
	if host == "" {
		// e.g. "api.deepseek.com/v1" parsed without scheme — Path holds it.
		host = strings.SplitN(strings.TrimPrefix(u.Path, "/"), "/", 2)[0]
	}
	return strings.EqualFold(host, deepSeekHost)
}

// isDeepSeek reports whether THIS provider instance is pointed at
// DeepSeek. A test seam (forceDeepSeek) lets internal wire tests exercise
// the DeepSeek path against a local httptest server whose host isn't
// api.deepseek.com.
func (p *Provider) isDeepSeek() bool {
	return p.forceDeepSeek || isDeepSeekEndpoint(p.baseURL)
}

// responseFormatFor lowers a bridle ResponseFormat to the OpenAI wire
// union, applying per-endpoint capability degradation.
//
// NEX-587: DeepSeek's /v1 REJECTS response_format type=json_schema
// (strict OR non-strict) with 400 "This response_format type is
// unavailable now" — discovered by the NEX-297 L2 live A/B. Rather than
// put wire on the line that's guaranteed to 400 every classifier call,
// the DeepSeek-aware provider degrades json_schema → json_object: looser
// (guarantees valid JSON, not schema-match) but portable, mirroring
// nexus's CheapModelFilter default. The harness/caller's tolerant
// JSON-parse path absorbs the shape-match loss. OpenAI-proper keeps
// strict json_schema unchanged (the NEX-300 promise).
func (p *Provider) responseFormatFor(rf *bridle.ResponseFormat) *openai.ChatCompletionNewParamsResponseFormatUnion {
	if rf == nil {
		return nil
	}
	if p.isDeepSeek() && rf.Type == "json_schema" {
		// Degrade to json_object — the portable shape DeepSeek supports.
		return toOpenAIResponseFormat(&bridle.ResponseFormat{Type: "json_object"})
	}
	return toOpenAIResponseFormat(rf)
}

// classifyOpenAIError maps an OpenAI-shape API error to a bridle
// ProviderError so callers can distinguish auth / rate-limit / server /
// timeout failures from a generic wrap. Returns nil when err is not an
// *openai.Error (e.g. context cancellation, dial failure) — RunTurn
// falls back to the generic wrap in that case.
//
// NEX-587: DeepSeek surfaces the standard OpenAI error envelope with an
// HTTP status code (429 on rate-limit, 401 on bad key, 5xx on overload),
// so the status-code mapping that works for OpenAI-proper covers DeepSeek
// too. The funnel's retry/backoff keys off ProviderErrorRateLimit; before
// this, every DeepSeek API error reached it as an opaque
// "openai: API error: ..." string with no kind.
func classifyOpenAIError(err error) *bridle.ProviderError {
	if err == nil {
		return nil
	}
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return nil
	}
	kind := kindForStatus(apiErr.StatusCode)
	return &bridle.ProviderError{
		Kind:    kind,
		Message: fmt.Sprintf("openai: API error (status %d)", apiErr.StatusCode),
		Err:     err,
	}
}

// kindForStatus maps an HTTP status code to a bridle ProviderErrorKind.
func kindForStatus(status int) bridle.ProviderErrorKind {
	switch status {
	case http.StatusTooManyRequests: // 429
		return bridle.ProviderErrorRateLimit
	case http.StatusUnauthorized, http.StatusForbidden: // 401, 403
		return bridle.ProviderErrorAuthFailed
	case http.StatusGatewayTimeout, http.StatusRequestTimeout: // 504, 408
		return bridle.ProviderErrorTimeout
	}
	if status >= 500 {
		return bridle.ProviderErrorServerError
	}
	// Other 4xx (400 bad-request, 404 model-not-found, 422): a typed but
	// non-specific server-reported error. Use ServerError so callers see
	// a ProviderError rather than a raw wrap; it is distinct from the
	// retryable rate-limit / transient 5xx via the message + status.
	return bridle.ProviderErrorServerError
}
