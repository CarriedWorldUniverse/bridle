package bridle

import (
	_ "embed"
	"fmt"
	"sort"
	"sync"

	"github.com/BurntSushi/toml"
)

//go:embed registry_models.toml
var catalogTOML []byte

// catalogFile is the TOML shape of registry_models.toml.
type catalogFile struct {
	Model []catalogModel `toml:"model"`
}

type catalogModel struct {
	Lane            string              `toml:"lane"`
	ID              string              `toml:"id"`
	Provider        string              `toml:"provider"`
	Aliases         []string            `toml:"aliases"`
	ContextWindow   int                 `toml:"context_window"`
	MaxOutputTokens int                 `toml:"max_output_tokens"`
	Capabilities    catalogCapabilities `toml:"capabilities"`
	Pricing         *catalogPricing     `toml:"pricing"`
	Prompt          *catalogPrompt      `toml:"prompt"`
}

type catalogCapabilities struct {
	Tools            bool   `toml:"tools"`
	ParallelTools    bool   `toml:"parallel_tools"`
	Streaming        bool   `toml:"streaming"`
	ReasoningEffort  bool   `toml:"reasoning_effort"`
	StructuredOutput bool   `toml:"structured_output"`
	PromptCaching    bool   `toml:"prompt_caching"`
	Vision           bool   `toml:"vision"`
	SystemPromptMode string `toml:"system_prompt_mode"`
}

type catalogPricing struct {
	In     float64 `toml:"in"`
	Out    float64 `toml:"out"`
	Cached float64 `toml:"cached"`
}

type catalogPrompt struct {
	Dialect      *catalogDialect `toml:"dialect"`
	RenditionRef string          `toml:"rendition_ref"`
}

type catalogDialect struct {
	ToolIdiom string `toml:"tool_idiom"`
	Format    string `toml:"format"`
	Thinking  string `toml:"thinking"`
}

// parseCatalog decodes the embedded TOML into ModelInfo values, keyed
// by the caller into the registry's "lane/id" map.
func parseCatalog(data []byte) ([]ModelInfo, error) {
	var f catalogFile
	if err := toml.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("bridle: parse registry_models.toml: %w", err)
	}
	out := make([]ModelInfo, 0, len(f.Model))
	for _, m := range f.Model {
		if m.Lane == "" || m.ID == "" {
			return nil, fmt.Errorf("bridle: registry_models.toml: model row missing lane or id (id=%q lane=%q)", m.ID, m.Lane)
		}
		// Fail-closed catalog-consistency guard (was previously only a
		// unit test, TestCatalog_StaticConsistency, which only covers
		// what's in registry_models.toml TODAY — not a future row a PR
		// adds). ContextWindow<=0 divides-by-zero / unbounds downstream
		// budget consumers (skills 2% budget, context-manager); an
		// invalid SystemPromptMode is a config typo. Both are build-time
		// bugs in registry_models.toml, so NewRegistry panics on them
		// (see loadCatalog/NewRegistry) exactly like a TOML parse error.
		if m.ContextWindow <= 0 {
			return nil, fmt.Errorf("bridle: registry_models.toml: %s/%s: context_window must be > 0, got %d", m.Lane, m.ID, m.ContextWindow)
		}
		mode := SystemPromptMode(m.Capabilities.SystemPromptMode)
		if !mode.IsValid() {
			return nil, fmt.Errorf("bridle: registry_models.toml: %s/%s: system_prompt_mode %q is not valid", m.Lane, m.ID, m.Capabilities.SystemPromptMode)
		}
		mi := ModelInfo{
			ID:              m.ID,
			Lane:            LaneID(m.Lane),
			Provider:        ProviderID(m.Provider),
			Aliases:         m.Aliases,
			ContextWindow:   m.ContextWindow,
			MaxOutputTokens: m.MaxOutputTokens,
			Capabilities: ModelCapabilities{
				Tools:            m.Capabilities.Tools,
				ParallelTools:    m.Capabilities.ParallelTools,
				Streaming:        m.Capabilities.Streaming,
				ReasoningEffort:  m.Capabilities.ReasoningEffort,
				StructuredOutput: m.Capabilities.StructuredOutput,
				PromptCaching:    m.Capabilities.PromptCaching,
				Vision:           m.Capabilities.Vision,
				SystemPromptMode: SystemPromptMode(m.Capabilities.SystemPromptMode),
			},
		}
		if m.Pricing != nil {
			mi.Pricing = &ModelPricing{In: m.Pricing.In, Out: m.Pricing.Out, Cached: m.Pricing.Cached}
		}
		if m.Prompt != nil {
			pm := &ModelPromptMeta{RenditionRef: m.Prompt.RenditionRef}
			if m.Prompt.Dialect != nil {
				pm.Dialect = &PromptDialect{
					ToolIdiom: m.Prompt.Dialect.ToolIdiom,
					Format:    m.Prompt.Dialect.Format,
					Thinking:  m.Prompt.Dialect.Thinking,
				}
			}
			mi.Prompt = pm
		}
		out = append(out, mi)
	}
	return out, nil
}

var (
	catalogOnce   sync.Once
	catalogModels []ModelInfo
	catalogErr    error
)

// loadCatalog parses the embedded registry_models.toml exactly once per
// process. The catalog is a compiled-in asset (go:embed) — a parse
// failure here is a build-time bug in registry_models.toml, not a
// runtime condition callers can recover from, so NewRegistry panics on
// it (see NewRegistry).
func loadCatalog() ([]ModelInfo, error) {
	catalogOnce.Do(func() {
		catalogModels, catalogErr = parseCatalog(catalogTOML)
	})
	return catalogModels, catalogErr
}

// catalogKey is the map key format for Registry.models: "lane/id".
func catalogKey(lane, id string) string {
	return lane + "/" + id
}

// ModelHandle is the resolved reference Registry.Resolve returns.
// Deliberately credential-free: Stream looks up the bound *Harness (and
// therefore the credentialed Provider) from the Handle's Lane at call
// time — the handle itself never carries auth/base-url configuration.
type ModelHandle struct {
	Lane     string
	Provider ProviderID
	Model    string
	Info     ModelInfo
}

// identityAliasKey is the map key for an identity-scoped alias
// registration: a STRUCT of (alias, identity), not a delimited string.
// Resolve/RegisterIdentityAlias previously built this key by
// concatenating alias+"@"+identity, which is NOT collision-proof:
// ("a@b", "c") and ("a", "b@c") both concatenate to "a@b@c" — a real
// risk since identities are sometimes email-shaped (contain "@"). A
// struct key compares both fields independently, so no delimiter
// ambiguity is possible regardless of what characters alias/identity
// contain.
type identityAliasKey struct {
	alias    string
	identity string
}

// Registry is bridle's facade over the static model catalog plus the
// caller-wired per-lane Harnesses. The catalog (registry_models.toml)
// NEVER constructs a Provider — Bind is the seam where deploy-specific
// credentials/base-URLs enter, wired by the caller once per lane.
type Registry struct {
	mu              sync.RWMutex
	models          map[string]ModelInfo        // keyed "lane/id" (catalogKey)
	aliases         map[string]string           // bare alias -> "lane/id"
	identityAliases map[identityAliasKey]string // (alias, identity) -> "lane/id"
	lanes           map[string]*Harness         // lane -> bound harness (from Bind)
}

// NewRegistry parses the embedded static catalog and returns an empty
// (unbound, alias-free) Registry. Panics if registry_models.toml itself
// fails to parse — see loadCatalog.
func NewRegistry() *Registry {
	models, err := loadCatalog()
	if err != nil {
		panic(err)
	}
	m := make(map[string]ModelInfo, len(models))
	for _, mi := range models {
		m[catalogKey(string(mi.Lane), mi.ID)] = mi
	}
	return &Registry{
		models:          m,
		aliases:         make(map[string]string),
		identityAliases: make(map[identityAliasKey]string),
		lanes:           make(map[string]*Harness),
	}
}

// Bind wires a credentialed Provider-backed Harness to a lane. The
// caller (agora/funnel deploy config) constructs the Harness with
// whatever creds/base-URL that lane's deploy needs; the registry never
// constructs Providers itself. Re-Binding a lane replaces its Harness.
func (r *Registry) Bind(lane string, h *Harness) error {
	if lane == "" {
		return fmt.Errorf("bridle: Bind: lane must not be empty")
	}
	if h == nil {
		return fmt.Errorf("bridle: Bind: harness must not be nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lanes[lane] = h
	return nil
}

// RegisterAlias maps a bare alias onto an already-cataloged "lane/id"
// target. Errors if the target isn't cataloged — aliasing to nothing is
// a config bug caught at registration time, not resolution time.
//
// For an identity-scoped override (agora's {identity}-interpolated
// alias form), use RegisterIdentityAlias instead — it takes alias and
// identity as SEPARATE parameters, keyed internally by a struct rather
// than a delimited "alias@identity" string, so it can't collide with a
// bare alias or another identity's override even when alias/identity
// values contain "@" (email-shaped identities).
func (r *Registry) RegisterAlias(alias, target string) error {
	if alias == "" {
		return fmt.Errorf("bridle: RegisterAlias: alias must not be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.models[target]; !ok {
		return fmt.Errorf("bridle: RegisterAlias(%q): target %q not cataloged", alias, target)
	}
	r.aliases[alias] = target
	return nil
}

// RegisterIdentityAlias maps an identity-scoped alias override — the
// same alias name resolving differently per-identity (agora's
// {identity}-interpolated alias form) — onto an already-cataloged
// "lane/id" target. Keyed by the (alias, identity) STRUCT, not a
// delimited string, so it is collision-proof regardless of what
// characters alias or identity contain (see identityAliasKey). Errors
// if the target isn't cataloged, same as RegisterAlias.
func (r *Registry) RegisterIdentityAlias(alias, identity, target string) error {
	if alias == "" {
		return fmt.Errorf("bridle: RegisterIdentityAlias: alias must not be empty")
	}
	if identity == "" {
		return fmt.Errorf("bridle: RegisterIdentityAlias: identity must not be empty (use RegisterAlias for a bare alias)")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.models[target]; !ok {
		return fmt.Errorf("bridle: RegisterIdentityAlias(%q, %q): target %q not cataloged", alias, identity, target)
	}
	r.identityAliases[identityAliasKey{alias: alias, identity: identity}] = target
	return nil
}

// Resolve implements the two-phase alias cascade: the identity-scoped
// override -> the bare alias -> bare "lane/id" -> error. Unresolvable
// input errors HERE, at call time — never mid-turn (agora-spec-bridle
// §1). A cataloged model whose lane hasn't been Bound also errors here,
// for the same reason: resolving to a handle nothing can Stream against
// is exactly the kind of failure that must surface at session/run start.
func (r *Registry) Resolve(aliasOrID, identity string) (ModelHandle, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	target := aliasOrID
	if identity != "" {
		if t, ok := r.identityAliases[identityAliasKey{alias: aliasOrID, identity: identity}]; ok {
			target = t
		} else if t, ok := r.aliases[aliasOrID]; ok {
			target = t
		}
	} else if t, ok := r.aliases[aliasOrID]; ok {
		target = t
	}

	info, ok := r.models[target]
	if !ok {
		return ModelHandle{}, fmt.Errorf("bridle: Resolve(%q): %q not cataloged", aliasOrID, target)
	}
	if _, bound := r.lanes[string(info.Lane)]; !bound {
		return ModelHandle{}, fmt.Errorf("bridle: Resolve(%q): lane %q not bound (call Bind first)", aliasOrID, info.Lane)
	}
	return ModelHandle{
		Lane:     string(info.Lane),
		Provider: info.Provider,
		Model:    info.ID,
		Info:     copyModelInfo(info), // see List's doc comment: don't alias the shared catalog
	}, nil
}

// List returns every cataloged ModelInfo, sorted by lane then id for
// deterministic output (feeds the TUI %-picker / /model per
// agora-spec-bridle §1). Each returned ModelInfo is a deep copy (see
// copyModelInfo) — the catalog's ModelInfo values are backed by a
// package-level sync.Once slice shared across EVERY Registry instance
// in the process, so without copying, a caller mutating
// List()[i].Pricing.In (a pointer) or appending to List()[i].Aliases
// would corrupt the catalog for every other Registry too.
func (r *Registry) List() []ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]ModelInfo, 0, len(r.models))
	for _, mi := range r.models {
		out = append(out, copyModelInfo(mi))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Lane != out[j].Lane {
			return out[i].Lane < out[j].Lane
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// copyModelInfo returns a deep copy of mi, so a caller mutating the
// returned value's nested pointers/slices (Pricing, Prompt, Aliases)
// cannot reach back into the shared catalog storage. See List's doc
// comment for why this matters.
func copyModelInfo(mi ModelInfo) ModelInfo {
	if mi.Aliases != nil {
		mi.Aliases = append([]string(nil), mi.Aliases...)
	}
	if mi.Pricing != nil {
		p := *mi.Pricing
		mi.Pricing = &p
	}
	if mi.Prompt != nil {
		pm := *mi.Prompt
		if mi.Prompt.Dialect != nil {
			d := *mi.Prompt.Dialect
			pm.Dialect = &d
		}
		mi.Prompt = &pm
	}
	return mi
}
