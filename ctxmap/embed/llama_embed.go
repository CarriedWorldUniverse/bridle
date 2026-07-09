//go:build ctxmap_llama

package embed

import (
	"fmt"
	"sync"

	"github.com/CarriedWorldUniverse/bridle/ctxmap/extractor/llama"
)

type Llama struct {
	mu  sync.Mutex
	m   *llama.Model
	ctx *llama.Context
}

func NewLlama(modelPath string, threads int) (*Llama, error) {
	m, err := llama.LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("embed model: %w", err)
	}
	ctx, err := m.NewEmbedContext(2048, threads)
	if err != nil {
		return nil, fmt.Errorf("embed context: %w", err)
	}
	return &Llama{m: m, ctx: ctx}, nil
}

func (l *Llama) Embed(text string) ([]float32, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	// nomic-embed expects a task prefix; facts are documents.
	return l.ctx.Embed("search_document: " + text)
}

func (l *Llama) Close() {
	l.ctx.Free()
	l.m.Free()
}

