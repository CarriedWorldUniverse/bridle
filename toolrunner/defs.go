package toolrunner

import (
	"encoding/json"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func obj(s string) json.RawMessage { return json.RawMessage(s) }

// Defs returns the JSON-schema tool surface for the local lane. The funnel
// passes these as TurnRequest.Tools alongside the LocalToolRunner.
func Defs() []bridle.ToolDef {
	return []bridle.ToolDef{
		{
			Name:        "bash",
			Description: "Run a shell command in the aspect's working directory. Returns stdout, stderr, exit_code.",
			InputSchema: obj(`{"type":"object","properties":{"command":{"type":"string","description":"the shell command"},"timeout_ms":{"type":"integer","description":"optional per-call timeout in ms"}},"required":["command"]}`),
		},
		{
			Name:        "read",
			Description: "Read a UTF-8 text file relative to the working directory.",
			InputSchema: obj(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
		},
		{
			Name:        "write",
			Description: "Create or overwrite a file relative to the working directory.",
			InputSchema: obj(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`),
		},
		{
			Name:        "edit",
			Description: "Replace old_string with new_string in a file. Fails if old_string is not unique unless replace_all is true.",
			InputSchema: obj(`{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]}`),
		},
		{
			Name:        "glob",
			Description: "Find files by glob pattern (supports a leading **/ for any depth). Returns paths relative to the working directory.",
			InputSchema: obj(`{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string","description":"optional subdirectory to search under"}},"required":["pattern"]}`),
		},
		{
			Name:        "grep",
			Description: "Search file contents by Go regular expression. Optionally restrict to files matching a glob.",
			InputSchema: obj(`{"type":"object","properties":{"pattern":{"type":"string"},"glob":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}`),
		},
		{
			Name:        "web_fetch",
			Description: "Fetch a web page as cleaned markdown (via the lynxai browser service).",
			InputSchema: obj(`{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}`),
		},
		{
			Name:        "web_extract",
			Description: "Extract structured JSON from a web page given a JSON schema (via lynxai).",
			InputSchema: obj(`{"type":"object","properties":{"url":{"type":"string"},"schema":{"type":"object"}},"required":["url","schema"]}`),
		},
	}
}
