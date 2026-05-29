package toolrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type webFetchArgs struct {
	URL string `json:"url"`
}
type webExtractArgs struct {
	URL    string          `json:"url"`
	Schema json.RawMessage `json:"schema"`
}

// lynxaiPost posts body to {base}{path} and returns the raw response bytes.
// HTTP/transport failures become a tool-level error result (so the model can react).
func (r *LocalToolRunner) lynxaiPost(ctx context.Context, path string, body any) (json.RawMessage, error) {
	if r.cfg.LynxaiBaseURL == "" {
		return result(map[string]string{"error": "web access disabled: no lynxai endpoint configured"})
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("lynxai: marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.cfg.LynxaiBaseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("lynxai: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if r.cfg.LynxaiKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.cfg.LynxaiKey)
	}
	resp, err := r.cfg.HTTPClient.Do(req)
	if err != nil {
		return result(map[string]string{"error": "lynxai unreachable: " + err.Error()})
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, int64(r.cfg.MaxOutputBytes)))
	if resp.StatusCode != http.StatusOK {
		return result(map[string]string{"error": fmt.Sprintf("lynxai %s: status %d: %s", path, resp.StatusCode, string(raw))})
	}
	return json.RawMessage(raw), nil
}

func (r *LocalToolRunner) runWebFetch(ctx context.Context, raw json.RawMessage) (json.RawMessage, error) {
	var a webFetchArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("web_fetch: bad args: %w", err)
	}
	return r.lynxaiPost(ctx, "/fetch", webFetchArgs{URL: a.URL})
}

func (r *LocalToolRunner) runWebExtract(ctx context.Context, raw json.RawMessage) (json.RawMessage, error) {
	var a webExtractArgs
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("web_extract: bad args: %w", err)
	}
	return r.lynxaiPost(ctx, "/extract", a)
}
