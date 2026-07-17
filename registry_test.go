package bridle

import "testing"

// coreLanes mirrors the 7 lanes registry_models.toml catalogs.
var coreLanes = []string{
	"claude-api", "claude-code", "claude-sdk", "openai", "deepseek",
	"gemini-api", "bedrock", "ollama",
}

func TestNewRegistry_CatalogHasCoreLanes(t *testing.T) {
	r := NewRegistry()
	models := r.List()
	if len(models) == 0 {
		t.Fatal("List() returned no models — catalog failed to load")
	}
	seen := make(map[string]bool)
	for _, mi := range models {
		seen[string(mi.Lane)] = true
	}
	for _, lane := range coreLanes {
		if !seen[lane] {
			t.Errorf("catalog missing lane %q", lane)
		}
	}
}

// TestCatalog_StaticConsistency asserts every cataloged model has a
// non-zero ContextWindow and a valid SystemPromptMode — the DoD's
// static-consistency check.
func TestCatalog_StaticConsistency(t *testing.T) {
	r := NewRegistry()
	for _, mi := range r.List() {
		if mi.ContextWindow <= 0 {
			t.Errorf("%s/%s: ContextWindow must be > 0, got %d", mi.Lane, mi.ID, mi.ContextWindow)
		}
		if !mi.Capabilities.SystemPromptMode.IsValid() {
			t.Errorf("%s/%s: SystemPromptMode %q is not valid", mi.Lane, mi.ID, mi.Capabilities.SystemPromptMode)
		}
	}
}

func TestRegistry_ResolveUncatalogedLane(t *testing.T) {
	r := NewRegistry()
	_, err := r.Resolve("nonexistent-lane/some-model", "")
	if err == nil {
		t.Fatal("expected error resolving an uncataloged lane/model, got nil")
	}
}

func TestRegistry_ResolveBareID_RequiresBind(t *testing.T) {
	r := NewRegistry()
	// Cataloged but unbound: Resolve must fail (never mid-turn).
	if _, err := r.Resolve("claude-api/claude-sonnet-5", ""); err == nil {
		t.Fatal("expected error resolving a cataloged-but-unbound lane, got nil")
	}

	if err := r.Bind("claude-api", NewHarness(nil)); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	handle, err := r.Resolve("claude-api/claude-sonnet-5", "")
	if err != nil {
		t.Fatalf("Resolve after Bind: %v", err)
	}
	if handle.Lane != "claude-api" || handle.Model != "claude-sonnet-5" {
		t.Errorf("unexpected handle: %+v", handle)
	}
	if handle.Provider != ProviderClaude {
		t.Errorf("expected Provider %q, got %q", ProviderClaude, handle.Provider)
	}
}

func TestRegistry_ResolveAliasCascade(t *testing.T) {
	r := NewRegistry()
	if err := r.Bind("claude-api", NewHarness(nil)); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	if err := r.RegisterAlias("main", "claude-api/claude-sonnet-5"); err != nil {
		t.Fatalf("RegisterAlias(main): %v", err)
	}
	if err := r.RegisterIdentityAlias("main", "shadow", "claude-api/claude-opus-4-8"); err != nil {
		t.Fatalf("RegisterIdentityAlias(main, shadow): %v", err)
	}

	// Phase 1: alias@identity wins over the bare alias when both exist.
	h, err := r.Resolve("main", "shadow")
	if err != nil {
		t.Fatalf("Resolve(main, shadow): %v", err)
	}
	if h.Model != "claude-opus-4-8" {
		t.Errorf("expected identity-scoped alias to win, got model %q", h.Model)
	}

	// Phase 2: bare alias, no identity match.
	h, err = r.Resolve("main", "someone-else")
	if err != nil {
		t.Fatalf("Resolve(main, someone-else): %v", err)
	}
	if h.Model != "claude-sonnet-5" {
		t.Errorf("expected bare-alias fallback, got model %q", h.Model)
	}

	// Phase 2b: bare alias, no identity at all.
	h, err = r.Resolve("main", "")
	if err != nil {
		t.Fatalf("Resolve(main, \"\"): %v", err)
	}
	if h.Model != "claude-sonnet-5" {
		t.Errorf("expected bare-alias resolution, got model %q", h.Model)
	}

	// Phase 3: bare lane/id, no alias registered for it.
	h, err = r.Resolve("claude-api/claude-fable-5", "")
	if err != nil {
		t.Fatalf("Resolve(bare lane/id): %v", err)
	}
	if h.Model != "claude-fable-5" {
		t.Errorf("expected bare lane/id resolution, got model %q", h.Model)
	}

	// Miss: nothing matches at any phase.
	if _, err := r.Resolve("no-such-alias", ""); err == nil {
		t.Fatal("expected error resolving an unregistered alias, got nil")
	}
}

func TestRegistry_RegisterAlias_RejectsUncatalogedTarget(t *testing.T) {
	r := NewRegistry()
	if err := r.RegisterAlias("bad", "claude-api/does-not-exist"); err == nil {
		t.Fatal("expected error registering an alias to an uncataloged target, got nil")
	}
}

func TestRegistry_RegisterIdentityAlias_RejectsUncatalogedTarget(t *testing.T) {
	r := NewRegistry()
	if err := r.RegisterIdentityAlias("bad", "someone", "claude-api/does-not-exist"); err == nil {
		t.Fatal("expected error registering an identity alias to an uncataloged target, got nil")
	}
}

// TestRegistry_IdentityAlias_NoCrossCollision proves the identity-scoped
// alias key is collision-proof: Resolve("a@b", "c") and Resolve("a",
// "b@c") must NOT be confused with each other, even though naively
// concatenating alias+"@"+identity produces the same string ("a@b@c")
// for both. Real risk: identities are sometimes email-shaped (contain
// "@"), so a naive string-concat key lets one identity's override
// resolve for a different (alias, identity) pair.
func TestRegistry_IdentityAlias_NoCrossCollision(t *testing.T) {
	r := NewRegistry()
	if err := r.Bind("claude-api", NewHarness(nil)); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	// Register "a@b" (as a literal alias, containing "@") scoped to
	// identity "c" -> opus.
	if err := r.RegisterIdentityAlias("a@b", "c", "claude-api/claude-opus-4-8"); err != nil {
		t.Fatalf("RegisterIdentityAlias(a@b, c): %v", err)
	}
	// Register "a" scoped to identity "b@c" (an email-shaped identity)
	// -> a DIFFERENT target (sonnet).
	if err := r.RegisterIdentityAlias("a", "b@c", "claude-api/claude-sonnet-5"); err != nil {
		t.Fatalf("RegisterIdentityAlias(a, b@c): %v", err)
	}

	h1, err := r.Resolve("a@b", "c")
	if err != nil {
		t.Fatalf("Resolve(a@b, c): %v", err)
	}
	if h1.Model != "claude-opus-4-8" {
		t.Errorf("Resolve(a@b, c) = %q, want claude-opus-4-8 (must not cross-resolve to the a/b@c registration)", h1.Model)
	}

	h2, err := r.Resolve("a", "b@c")
	if err != nil {
		t.Fatalf("Resolve(a, b@c): %v", err)
	}
	if h2.Model != "claude-sonnet-5" {
		t.Errorf("Resolve(a, b@c) = %q, want claude-sonnet-5 (must not cross-resolve to the a@b/c registration)", h2.Model)
	}
}

func TestRegistry_Bind_RejectsEmptyLaneOrNilHarness(t *testing.T) {
	r := NewRegistry()
	if err := r.Bind("", NewHarness(nil)); err == nil {
		t.Fatal("expected error binding an empty lane, got nil")
	}
	if err := r.Bind("claude-api", nil); err == nil {
		t.Fatal("expected error binding a nil harness, got nil")
	}
}

// TestParseCatalog_RejectsZeroContextWindow proves parseCatalog fails
// closed (an error, not a silently-accepted zero) when a row's
// context_window is missing/zero. Downstream consumers (skills 2%
// budget, context-manager) divide-by-zero or treat the budget as
// unbounded on a zero ContextWindow — this must be caught at parse
// time, not left to a separate unit test (TestCatalog_StaticConsistency)
// that only covers today's embedded catalog contents.
func TestParseCatalog_RejectsZeroContextWindow(t *testing.T) {
	data := []byte(`
[[model]]
lane = "claude-api"
id = "bad-model"
provider = "claude-api"
[model.capabilities]
system_prompt_mode = "append"
`)
	if _, err := parseCatalog(data); err == nil {
		t.Fatal("expected an error for a row with context_window omitted (0), got nil")
	}
}

// TestParseCatalog_RejectsInvalidSystemPromptMode proves parseCatalog
// fails closed on an unrecognized system_prompt_mode value.
func TestParseCatalog_RejectsInvalidSystemPromptMode(t *testing.T) {
	data := []byte(`
[[model]]
lane = "claude-api"
id = "bad-model"
provider = "claude-api"
context_window = 100000
[model.capabilities]
system_prompt_mode = "bogus"
`)
	if _, err := parseCatalog(data); err == nil {
		t.Fatal("expected an error for an invalid system_prompt_mode, got nil")
	}
}

// TestRegistry_List_DefensiveCopy proves a caller mutating a ModelInfo's
// nested pointer/slice fields returned from List() does NOT corrupt the
// shared catalog for other Registry instances — the catalog's
// ModelInfo values are backed by a package-level sync.Once slice shared
// across every Registry, so a caller mutating List()[i].Pricing.In (a
// pointer) previously corrupted the catalog process-wide.
func TestRegistry_List_DefensiveCopy(t *testing.T) {
	r1 := NewRegistry()
	models := r1.List()
	var target *ModelInfo
	for i := range models {
		if models[i].Pricing != nil {
			target = &models[i]
			break
		}
	}
	if target == nil {
		t.Fatal("no cataloged model has Pricing set — can't exercise the mutation")
	}
	original := target.Pricing.In
	target.Pricing.In = -999999 // mutate the pointer's target in place

	r2 := NewRegistry()
	for _, mi := range r2.List() {
		if mi.ID == target.ID && mi.Lane == target.Lane {
			if mi.Pricing.In == -999999 {
				t.Fatalf("mutating r1.List()'s ModelInfo.Pricing corrupted r2's catalog (want unaffected %v, got -999999)", original)
			}
		}
	}
}

func TestRegistry_List_Sorted(t *testing.T) {
	r := NewRegistry()
	models := r.List()
	for i := 1; i < len(models); i++ {
		prev, cur := models[i-1], models[i]
		if prev.Lane > cur.Lane || (prev.Lane == cur.Lane && prev.ID > cur.ID) {
			t.Fatalf("List() not sorted: %s/%s before %s/%s", prev.Lane, prev.ID, cur.Lane, cur.ID)
		}
	}
}
