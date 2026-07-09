//go:build ctxmap_llama

package embed

// Threshold calibration: labeled pairs of fact statements in three classes —
// DUP (same fact, different phrasing), CONTRA (same topic, incompatible
// content), UNREL (different facts). Run with -run Calibrate -v to print the
// cosine distributions; the reconciler thresholds are chosen from this data.
// Requires the nomic model on disk; skipped otherwise (integration test).

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
)

type pair struct{ a, b string }

var dupPairs = []pair{
	{"The render seat moves to ember-node.", "The render seat is now on the box called ember-node."},
	{"Bench results are stored in Postgres shared with ledger.", "Bench results go to the shared ledger Postgres database."},
	{"The settlement generator caps at 40 buildings per site.", "Settlement generation has a hard cap of 40 buildings per site."},
	{"Pressure buffers are preallocated at chunk load.", "The water sim preallocates its pressure buffers when a chunk loads."},
	{"Ornith runs on vLLM behind litellm.", "Ornith is served by vLLM with litellm in front."},
	{"Broadcasts break agent context and focus.", "Broadcast messages destroy the agents' context and focus."},
}

var contraPairs = []pair{
	{"Bench results are stored in SQLite.", "Bench results are stored in Postgres."},
	{"The render seat moves to ember-node.", "The render seat moves to forge-node."},
	{"The settlement generator caps at 40 buildings per site.", "The settlement generator caps at 12 buildings per site."},
	{"Biome blending must never sample across chunk seams.", "Biome blending samples freely across chunk seams."},
	{"The belief field write pass runs on aurora-queue.", "The belief field write pass runs on cinder-queue."},
	{"Keel is always-on.", "Keel is hard-locked off."},
}

var unrelPairs = []pair{
	{"The render seat moves to ember-node.", "Bench results are stored in Postgres."},
	{"The settlement generator caps at 40 buildings per site.", "Ornith runs on vLLM behind litellm."},
	{"Pressure buffers are preallocated at chunk load.", "The operator prefers economy readouts as tables."},
	{"Biome blending must never sample across chunk seams.", "Pool workers are named personality-role."},
	{"Maren schedules Meshy jobs overnight.", "The waterfall stutter is caused by the chunk loader."},
	{"The quant floor for extraction models is q8.", "Broadcasts break agent context and focus."},
}

func TestCalibrate(t *testing.T) {
	modelPath := filepath.Join(os.Getenv("HOME"), "models/nomic-embed-text-v1.5.Q8_0.gguf")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("embedding model not present")
	}
	e, err := NewLlama(modelPath, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	score := func(ps []pair) []float64 {
		var out []float64
		for _, p := range ps {
			va, err := e.Embed(p.a)
			if err != nil {
				t.Fatal(err)
			}
			vb, err := e.Embed(p.b)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, Cos(va, vb))
		}
		sort.Float64s(out)
		return out
	}

	dup := score(dupPairs)
	contra := score(contraPairs)
	unrel := score(unrelPairs)
	t.Logf("DUP    cosines: %v", dup)
	t.Logf("CONTRA cosines: %v", contra)
	t.Logf("UNREL  cosines: %v", unrel)

	// sanity ordering: min(dup) should exceed max(unrel); contra sits between.
	if dup[0] <= unrel[len(unrel)-1] {
		t.Errorf("dup/unrel distributions overlap: min(dup)=%.3f max(unrel)=%.3f", dup[0], unrel[len(unrel)-1])
	}
}
