package mcpclient_test

import (
	"context"
	"encoding/json"
	"testing"

	mcplib "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	"github.com/CarriedWorldUniverse/bridle/internal/mcpclient"
)

// newInProcessClient wires an MCPServer to an in-process client.
// Used for testing only — no subprocess, no network.
func newInProcessClient(srv *server.MCPServer) (*mcplib.Client, error) {
	return mcplib.NewInProcessClient(srv)
}

// newTestServer creates a minimal MCP server with one tool: echo_tool.
func newTestServer() *server.MCPServer {
	srv := server.NewMCPServer("test-server", "0.1")
	srv.AddTool(
		mcp.NewToolWithRawSchema("echo_tool", "Returns its input", json.RawMessage(`{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}`)),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			msg := req.GetString("msg", "")
			return &mcp.CallToolResult{
				Content: []mcp.Content{mcp.NewTextContent("echo: " + msg)},
			}, nil
		},
	)
	return srv
}

// connectInProcess builds a Client wrapping an already-initialized in-process client.
// This bypasses the Connect() transport factory for unit testing.
func connectInProcess(ctx context.Context, t *testing.T, srv *server.MCPServer) *mcpclient.Client {
	t.Helper()
	// Use the exported testing helper on Client — since we can't do that without
	// exposing internals, we run against a stdio fake server instead. But mark3labs
	// provides NewInProcessClient that we can use directly. The workaround is to
	// run a real stdio subprocess — or, simpler, test via the public interface
	// through a real server that speaks stdio. For unit tests we use a different
	// approach: test the mcpclient package via a thin stdio server wrapper.
	// Actually the cleanest: use server.NewTestServer() if it exists, or use
	// the InProcess path by writing a TestClient constructor.
	t.Skip("in-process path not yet exposed — tested via harness integration tests")
	return nil
}

// TestConnect_EmptyConfig verifies that Connect with nil/empty specs returns a no-op client.
func TestConnect_EmptyConfig(t *testing.T) {
	ctx := context.Background()
	c, err := mcpclient.Connect(ctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer c.Close()
	if len(c.Tools()) != 0 {
		t.Errorf("expected no tools, got %v", c.Tools())
	}
	if c.IsMCPTool("anything") {
		t.Error("IsMCPTool should return false on empty client")
	}
}

// TestConnect_UnknownTransport verifies that an unknown transport returns an error.
func TestConnect_UnknownTransport(t *testing.T) {
	ctx := context.Background()
	_, err := mcpclient.Connect(ctx, []mcpclient.ServerSpec{{
		Name:      "bad",
		Transport: "grpc",
	}})
	if err == nil {
		t.Fatal("expected error for unknown transport, got nil")
	}
}
