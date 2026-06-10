package bridle

import "testing"

// TestEstimateTokens_PositiveForNonEmpty — the estimator must never
// return zero for non-empty text; it is the floor that guarantees a
// completed turn never reports silently-zero usage.
func TestEstimateTokens_PositiveForNonEmpty(t *testing.T) {
	if got := estimateTokens("hello world"); got <= 0 {
		t.Fatalf("estimateTokens(non-empty) = %d, want > 0", got)
	}
	if got := estimateTokens(""); got != 0 {
		t.Fatalf("estimateTokens(\"\") = %d, want 0", got)
	}
}

// TestEstimateTokens_Deterministic — same input, same output. The
// estimator is a pure function of its text so usage is reproducible.
func TestEstimateTokens_Deterministic(t *testing.T) {
	s := "the quick brown fox jumps over the lazy dog"
	if estimateTokens(s) != estimateTokens(s) {
		t.Fatal("estimateTokens is not deterministic")
	}
}

// TestEstimateTokens_MonotonicWithLength — longer text estimates more
// tokens. This is the documented approximation contract: the estimate
// grows with input size so a big prompt never under-floors a small one.
func TestEstimateTokens_Monotonic(t *testing.T) {
	short := estimateTokens("a short prompt")
	long := estimateTokens("a short prompt with substantially more words tacked on to make it longer than the short one")
	if long <= short {
		t.Fatalf("estimate not monotonic: short=%d long=%d", short, long)
	}
}

// TestNormalizeUsage_ZeroGetsEstimated — a provider returning zero
// usage is the hole the contract closes: after normalization the usage
// is non-zero and flagged Estimated, with plausible (>0) counts derived
// from the prompt and response text.
func TestNormalizeUsage_ZeroGetsEstimated(t *testing.T) {
	got := normalizeUsage(Usage{}, "this is the prompt the model received", "and this is what it said back")
	if got.InputTokens <= 0 {
		t.Errorf("InputTokens = %d, want > 0 (estimated floor)", got.InputTokens)
	}
	if got.OutputTokens <= 0 {
		t.Errorf("OutputTokens = %d, want > 0 (estimated floor)", got.OutputTokens)
	}
	if !got.Estimated {
		t.Error("Estimated = false, want true when counts were filled by estimation")
	}
}

// TestNormalizeUsage_RealUsagePassesThrough — when the engine reported
// real counts, normalization is a no-op: values are untouched and
// Estimated stays false.
func TestNormalizeUsage_RealUsagePassesThrough(t *testing.T) {
	real := Usage{InputTokens: 123, OutputTokens: 45}
	got := normalizeUsage(real, "prompt text", "response text")
	if got.InputTokens != 123 || got.OutputTokens != 45 {
		t.Errorf("real usage mutated: %+v", got)
	}
	if got.Estimated {
		t.Error("Estimated = true, want false when engine reported real usage")
	}
}

// TestNormalizeUsage_PartialInputOnlyEstimatesOutput — some engines
// report input tokens but not output (or vice versa). Normalization
// fills ONLY the missing side and flags the result estimated; the
// reported side is preserved exactly.
func TestNormalizeUsage_PartialFillsMissingSide(t *testing.T) {
	got := normalizeUsage(Usage{InputTokens: 100}, "prompt", "a longer response than the prompt was")
	if got.InputTokens != 100 {
		t.Errorf("reported InputTokens mutated: got %d, want 100", got.InputTokens)
	}
	if got.OutputTokens <= 0 {
		t.Errorf("OutputTokens = %d, want > 0 (estimated)", got.OutputTokens)
	}
	if !got.Estimated {
		t.Error("Estimated = false, want true when one side was estimated")
	}
}

// TestNormalizeUsage_EmptyTextStaysZero — a genuinely empty turn (no
// prompt, no response) has nothing to estimate from; normalization
// leaves it zero and does NOT spuriously flag Estimated. The contract
// is "never silently zero a turn that had content", not "invent tokens
// for an empty turn".
func TestNormalizeUsage_EmptyTextStaysZero(t *testing.T) {
	got := normalizeUsage(Usage{}, "", "")
	if got.InputTokens != 0 || got.OutputTokens != 0 {
		t.Errorf("empty turn estimated non-zero: %+v", got)
	}
	if got.Estimated {
		t.Error("Estimated = true for an empty turn, want false")
	}
}
