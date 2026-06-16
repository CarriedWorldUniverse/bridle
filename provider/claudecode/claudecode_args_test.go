package claudecode

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

func TestBuildCLIArgs_AllowedTools(t *testing.T) {
	tests := []struct {
		name             string
		mcp              *bridle.MCPClientConfig
		tools            []bridle.ToolDef
		providerAllowed  []string
		providerDisallow []string
		model            string
		sessionID        string
		sessionNew       bool
		extraArgs        []string

		wantAllowedTools     string // comma-separated allowedTools, "" means flag absent
		wantDisallowedTools  bool
		wantModel            string
		wantSessionFlag      string // "--session-id" or "--resume" or ""
		wantExtraArg         string
	}{
		{
			name:        "MCP nil, no tools → no --allowedTools",
			mcp:         nil,
			tools:       nil,
			wantAllowedTools: "",
		},
		{
			name: "MCP nil, tools with names → --allowedTools from req.Tools",
			mcp:  nil,
			tools: []bridle.ToolDef{
				{Name: "send_chat"},
				{Name: "triage"},
			},
			wantAllowedTools: "send_chat,triage",
		},
		{
			name: "MCP non-nil, no tools → no --allowedTools (MCP discovery handles scoping)",
			mcp:  &bridle.MCPClientConfig{},
			tools: nil,
			wantAllowedTools: "",
		},
		{
			name: "MCP non-nil, tools with names → no --allowedTools (req.Tools must not block MCP-discovered tools)",
			mcp:  &bridle.MCPClientConfig{},
			tools: []bridle.ToolDef{
				{Name: "send_chat"},
				{Name: "triage"},
			},
			wantAllowedTools: "",
		},
		{
			name:            "MCP nil, no req.Tools → falls back to provider AllowedTools",
			mcp:             nil,
			tools:           nil,
			providerAllowed: []string{"Bash", "Read", "Write"},
			wantAllowedTools: "Bash,Read,Write",
		},
		{
			name: "MCP nil, req.Tools with empty-names only → falls back to provider AllowedTools (guard)",
			mcp:  nil,
			tools: []bridle.ToolDef{
				{Name: ""}, // degenerate — no usable names
			},
			providerAllowed:  []string{"Bash", "Read"},
			wantAllowedTools: "Bash,Read",
		},
		{
			name:    "MCP nil, req.Tools single tool with empty name → guard fallback",
			mcp:     nil,
			tools:   []bridle.ToolDef{{Name: ""}},
			// providerAllowed is empty too → no --allowedTools
			wantAllowedTools: "",
		},
		{
			name: "MCP non-nil with provider AllowedTools → still no --allowedTools (MCP takes precedence)",
			mcp:  &bridle.MCPClientConfig{},
			tools: nil,
			providerAllowed: []string{"Bash", "Read"},
			wantAllowedTools: "",
		},
		{
			name:                "DisallowedTools passed regardless of MCP",
			mcp:                 &bridle.MCPClientConfig{},
			tools:               nil,
			providerDisallow:    []string{"Task"},
			wantAllowedTools:    "",
			wantDisallowedTools: true,
		},
		{
			name:    "Model flag passed through",
			model:   "claude-opus-4-7",
			wantModel: "claude-opus-4-7",
		},
		{
			name:            "Session-id when New=true",
			sessionID:       "abc-123",
			sessionNew:      true,
			wantSessionFlag: "--session-id",
		},
		{
			name:            "Resume when New=false",
			sessionID:       "abc-123",
			sessionNew:      false,
			wantSessionFlag: "--resume",
		},
		{
			name:          "ExtraArgs appended",
			extraArgs:     []string{"--debug"},
			wantExtraArg:  "--debug",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := New()
			p.AllowedTools = tt.providerAllowed
			p.DisallowedTools = tt.providerDisallow
			p.ExtraArgs = tt.extraArgs

			req := bridle.ProviderRequest{
				MCP:   tt.mcp,
				Tools: tt.tools,
				Model: tt.model,
				Session: bridle.SessionHandle{
					ID:  tt.sessionID,
					New: tt.sessionNew,
				},
			}

			args, _, err := p.buildCLIArgs(req, tt.sessionNew)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Check --allowedTools presence/value.
			allowedIdx := indexOf(args, "--allowedTools")
			if tt.wantAllowedTools == "" {
				if allowedIdx >= 0 {
					t.Errorf("--allowedTools should be absent, got value %q", args[allowedIdx+1])
				}
			} else {
				if allowedIdx < 0 {
					t.Errorf("--allowedTools should be present with %q, but flag not found in %v", tt.wantAllowedTools, args)
				} else if allowedIdx+1 >= len(args) || args[allowedIdx+1] != tt.wantAllowedTools {
					got := ""
					if allowedIdx+1 < len(args) {
						got = args[allowedIdx+1]
					}
					t.Errorf("--allowedTools = %q, want %q", got, tt.wantAllowedTools)
				}
			}

			// Check --disallowedTools presence.
			disallowIdx := indexOf(args, "--disallowedTools")
			if tt.wantDisallowedTools && disallowIdx < 0 {
				t.Error("--disallowedTools should be present")
			}
			if !tt.wantDisallowedTools && disallowIdx >= 0 {
				t.Error("--disallowedTools should be absent")
			}

			// Check --model.
			modelIdx := indexOf(args, "--model")
			if tt.wantModel != "" {
				if modelIdx < 0 {
					t.Errorf("--model should be present with %q", tt.wantModel)
				} else if modelIdx+1 >= len(args) || args[modelIdx+1] != tt.wantModel {
					t.Errorf("--model = %q, want %q", args[modelIdx+1], tt.wantModel)
				}
			} else if modelIdx >= 0 {
				t.Error("--model should be absent")
			}

			// Check session flag.
			if tt.wantSessionFlag != "" {
				if indexOf(args, tt.wantSessionFlag) < 0 {
					t.Errorf("%s flag not found in %v", tt.wantSessionFlag, args)
				}
			}

			// Check extra args.
			if tt.wantExtraArg != "" {
				found := false
				for _, a := range args {
					if a == tt.wantExtraArg {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("extra arg %q not found in %v", tt.wantExtraArg, args)
				}
			}
		})
	}
}

func TestBuildCLIArgs_BaseArgsAlwaysPresent(t *testing.T) {
	p := New()
	req := bridle.ProviderRequest{}
	args, _, err := p.buildCLIArgs(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	required := []string{"-p", "--output-format", "stream-json", "--verbose", "--permission-mode", "bypassPermissions"}
	for _, flag := range required {
		if indexOf(args, flag) < 0 {
			t.Errorf("required flag %q not found in args", flag)
		}
	}

	// -p should be followed by a prompt argument (empty string is valid for no user message).
	pIdx := indexOf(args, "-p")
	if pIdx < 0 || pIdx+1 >= len(args) {
		t.Error("-p flag should have a following argument")
	}
}

func TestBuildCLIArgs_MCPWithServers(t *testing.T) {
	// Verify that a fully-populated MCP config still suppresses --allowedTools.
	p := New()
	p.AllowedTools = []string{"Bash", "Read"}

	req := bridle.ProviderRequest{
		MCP: &bridle.MCPClientConfig{
			Servers: []bridle.MCPServerSpec{
				{
					Name:      "nexus-jira",
					Transport: bridle.MCPTransportStdio,
					Command:   []string{"nexus-jira-mcp"},
				},
				{
					Name:      "nexus-comms",
					Transport: bridle.MCPTransportHTTPSSE,
					URL:       "http://localhost:9999/sse",
				},
			},
		},
		Tools: []bridle.ToolDef{
			{Name: "send_chat"},
			{Name: "triage"},
		},
		Model: "claude-opus-4-7",
	}

	args, _, err := p.buildCLIArgs(req, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if indexOf(args, "--allowedTools") >= 0 {
		t.Error("--allowedTools should be absent when MCP is configured with servers")
	}

	// Model should still pass through.
	if indexOf(args, "--model") < 0 {
		t.Error("--model should still be present")
	}

	// --mcp-config NOT passed when Cwd is empty (no path to resolve).
	if indexOf(args, "--mcp-config") >= 0 {
		t.Error("--mcp-config should not be passed when Cwd is empty")
	}
}

func TestBuildCLIArgs_MCPConfigFlag(t *testing.T) {
	t.Run("MCP set with Cwd and .mcp.json present → --mcp-config added", func(t *testing.T) {
		dir := t.TempDir()
		mcpPath := filepath.Join(dir, ".mcp.json")
		if err := os.WriteFile(mcpPath, []byte(`{"mcpServers":{}}`), 0644); err != nil {
			t.Fatal(err)
		}

		p := New()
		req := bridle.ProviderRequest{
			MCP: &bridle.MCPClientConfig{},
			Cwd: dir,
		}
		args, _, err := p.buildCLIArgs(req, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		idx := indexOf(args, "--mcp-config")
		if idx < 0 {
			t.Error("--mcp-config should be present when MCP is set and .mcp.json exists")
		} else if idx+1 >= len(args) || args[idx+1] != mcpPath {
			got := ""
			if idx+1 < len(args) {
				got = args[idx+1]
			}
			t.Errorf("--mcp-config path = %q, want %q", got, mcpPath)
		}

		// --allowedTools must still be suppressed.
		if indexOf(args, "--allowedTools") >= 0 {
			t.Error("--allowedTools should be absent when MCP is configured")
		}
	})

	t.Run("MCP set with Cwd but .mcp.json absent → --mcp-config skipped", func(t *testing.T) {
		dir := t.TempDir()
		// Don't create .mcp.json in this dir.

		p := New()
		req := bridle.ProviderRequest{
			MCP: &bridle.MCPClientConfig{},
			Cwd: dir,
		}
		args, _, err := p.buildCLIArgs(req, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if indexOf(args, "--mcp-config") >= 0 {
			t.Error("--mcp-config should be absent when .mcp.json does not exist")
		}
	})

	t.Run("MCP nil with Cwd and .mcp.json present → --mcp-config absent (backward compat)", func(t *testing.T) {
		dir := t.TempDir()
		mcpPath := filepath.Join(dir, ".mcp.json")
		if err := os.WriteFile(mcpPath, []byte(`{"mcpServers":{}}`), 0644); err != nil {
			t.Fatal(err)
		}

		p := New()
		p.AllowedTools = []string{"Bash"}
		req := bridle.ProviderRequest{
			MCP:   nil,
			Cwd:   dir,
			Tools: []bridle.ToolDef{{Name: "Read"}},
		}
		args, _, err := p.buildCLIArgs(req, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if indexOf(args, "--mcp-config") >= 0 {
			t.Error("--mcp-config should be absent when MCP is nil")
		}
		// --allowedTools should be present since MCP is nil.
		if indexOf(args, "--allowedTools") < 0 {
			t.Error("--allowedTools should be present when MCP is nil")
		}
	})
}

func TestBuildCLIArgs_SystemPromptInline(t *testing.T) {
	p := New()
	req := bridle.ProviderRequest{
		AppendSystemPrompt: "You are a helpful assistant.",
	}

	args, spillFile, err := p.buildCLIArgs(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spillFile != "" {
		t.Errorf("short system prompt should be inlined, not spilled to %q", spillFile)
	}

	inlineIdx := indexOf(args, "--append-system-prompt")
	fileIdx := indexOf(args, "--append-system-prompt-file")
	if inlineIdx < 0 {
		t.Error("--append-system-prompt should be present for short prompt")
	}
	if fileIdx >= 0 {
		t.Error("--append-system-prompt-file should not be present for short prompt")
	}
}

func TestBuildCLIArgs_SystemPromptReplace(t *testing.T) {
	p := New()
	req := bridle.ProviderRequest{
		AppendSystemPrompt: "You are a skilled Go developer.",
		SystemPromptMode:   bridle.SystemPromptReplace,
	}

	args, spillFile, err := p.buildCLIArgs(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spillFile != "" {
		t.Errorf("short system prompt in replace mode should be inlined")
	}

	inlineIdx := indexOf(args, "--system-prompt")
	fileIdx := indexOf(args, "--system-prompt-file")
	if inlineIdx < 0 || args[inlineIdx+1] != "You are a skilled Go developer." {
		t.Error("--system-prompt should be present for short prompt in replace mode")
	}
	if fileIdx >= 0 {
		t.Error("--system-prompt-file should not be present for short prompt in replace mode")
	}
}

func TestBuildCLIArgs_SystemPromptReplaceSpill(t *testing.T) {
	p := New()
	big := strings.Repeat("x", systemPromptSpillThresholdBytes+1)
	req := bridle.ProviderRequest{
		AppendSystemPrompt: big,
		SystemPromptMode:   bridle.SystemPromptReplace,
	}

	args, spillFile, err := p.buildCLIArgs(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spillFile == "" {
		t.Error("large system prompt in replace mode should spill to tempfile")
	}

	inlineIdx := indexOf(args, "--system-prompt")
	fileIdx := indexOf(args, "--system-prompt-file")
	if inlineIdx >= 0 {
		t.Error("--system-prompt should NOT be present for large prompt in replace mode")
	}
	if fileIdx < 0 {
		t.Error("--system-prompt-file should be present for large prompt in replace mode")
	}
}

func TestBuildCLIArgs_DefaultAppendMode(t *testing.T) {
	p := New()
	req := bridle.ProviderRequest{
		AppendSystemPrompt: "You are a helpful assistant.",
		// SystemPromptMode unspecified (zero value) = append.
	}

	args, spillFile, err := p.buildCLIArgs(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spillFile != "" {
		t.Errorf("short system prompt in default mode should be inlined")
	}

	inlineIdx := indexOf(args, "--append-system-prompt")
	fileIdx := indexOf(args, "--append-system-prompt-file")
	if inlineIdx < 0 || args[inlineIdx+1] != "You are a helpful assistant." {
		t.Error("--append-system-prompt should be present for short prompt in default mode")
	}
	if fileIdx >= 0 {
		t.Error("--append-system-prompt-file should not be present for short prompt in default mode")
	}
}

func TestBuildCLIArgs_Bare(t *testing.T) {
	p := New()
	p.Bare = true

	args, _, err := p.buildCLIArgs(bridle.ProviderRequest{}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if indexOf(args, "--bare") < 0 {
		t.Error("--bare should be present when Bare=true")
	}
}

func indexOf(slice []string, s string) int {
	for i, v := range slice {
		if v == s {
			return i
		}
	}
	return -1
}
