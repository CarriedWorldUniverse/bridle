package codexcli

import (
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func TestMCPConfigArgs(t *testing.T) {
	got := mcpConfigArgs([]bridle.MCPServerSpec{
		{Name: "nexus-comms", Transport: bridle.MCPTransportStdio,
			Command: []string{"nexus-comms-mcp", "--keyfile", "/k.json"},
			Env:     map[string]string{"NEXUS_URL": "wss://x:7888/connect", "B": "2"}},
		{Name: "remote", Transport: bridle.MCPTransportHTTPSSE, URL: "https://h/sse"},
	})
	joined := strings.Join(got, " ")
	t.Logf("ARGS: %s", joined)
	for _, want := range []string{
		`mcp_servers.nexus-comms.command="nexus-comms-mcp"`,
		`mcp_servers.nexus-comms.args=["--keyfile","/k.json"]`,
		`mcp_servers.nexus-comms.env={B="2",NEXUS_URL="wss://x:7888/connect"}`,
		`mcp_servers.remote.url="https://h/sse"`,
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q", want)
		}
	}
}
