// Package embed defines the sentence-embedding seam for the reconciler.
// The llama.cpp-backed implementation is behind the ctxmap_llama build tag.
package embed

// Embedder produces L2-normalized sentence vectors.
type Embedder interface {
	Embed(text string) ([]float32, error)
}

// Cos computes cosine similarity. Inputs are already L2-normalized by
// conforming implementations, so this is a plain dot product.
func Cos(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}
