//go:build ctxmap_llama

// Package llama is a minimal cgo binding against llama.cpp's C API,
// sized for exactly one job: schema-constrained fact extraction on CPU.
// Spike 1 of ctxmap — proves the in-process-extractor path inside a Go harness.
package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../../../vendor-llama/llama.cpp/include -I${SRCDIR}/../../../vendor-llama/llama.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../../vendor-llama/llama.cpp/lib -lllama -lggml -lggml-base -lggml-cpu -lm -lstdc++ -Wl,-rpath,${SRCDIR}/../../../vendor-llama/llama.cpp/lib

#include <stdlib.h>
#include "llama.h"

// small C shims where the API wants struct-by-value defaults
static struct llama_model_params spike_model_params(void) {
	struct llama_model_params p = llama_model_default_params();
	p.n_gpu_layers = 0; // CPU only, by design
	return p;
}
static struct llama_context_params spike_ctx_params(int n_ctx, int n_threads) {
	struct llama_context_params p = llama_context_default_params();
	p.n_ctx = n_ctx;
	p.n_threads = n_threads;
	p.n_threads_batch = n_threads;
	return p;
}
static struct llama_context_params spike_embed_ctx_params(int n_ctx, int n_threads) {
	struct llama_context_params p = llama_context_default_params();
	p.n_ctx = n_ctx;
	p.n_threads = n_threads;
	p.n_threads_batch = n_threads;
	p.embeddings = true;
	p.pooling_type = LLAMA_POOLING_TYPE_MEAN;
	return p;
}
*/
import "C"

import (
	"fmt"
	"math"
	"strings"
	"unsafe"
)

type Model struct {
	model *C.struct_llama_model
	vocab *C.struct_llama_vocab
}

type Context struct {
	ctx  *C.struct_llama_context
	m    *Model
	nCtx int
}

func LoadModel(path string) (*Model, error) {
	C.llama_backend_init()
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	m := C.llama_model_load_from_file(cpath, C.spike_model_params())
	if m == nil {
		return nil, fmt.Errorf("failed to load model %s", path)
	}
	return &Model{model: m, vocab: C.llama_model_get_vocab(m)}, nil
}

func (m *Model) NewContext(nCtx, nThreads int) (*Context, error) {
	ctx := C.llama_init_from_model(m.model, C.spike_ctx_params(C.int(nCtx), C.int(nThreads)))
	if ctx == nil {
		return nil, fmt.Errorf("failed to create context")
	}
	return &Context{ctx: ctx, m: m, nCtx: nCtx}, nil
}

func (m *Model) tokenize(text string, addBOS bool) ([]C.llama_token, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	maxTokens := len(text) + 8
	buf := make([]C.llama_token, maxTokens)
	n := C.llama_tokenize(m.vocab, ctext, C.int32_t(len(text)),
		&buf[0], C.int32_t(maxTokens), C.bool(addBOS), C.bool(true))
	if n < 0 {
		return nil, fmt.Errorf("tokenize failed: need %d tokens", -n)
	}
	return buf[:n], nil
}

func (m *Model) detokenize(tok C.llama_token) string {
	var buf [256]C.char
	n := C.llama_token_to_piece(m.vocab, tok, &buf[0], 256, 0, C.bool(true))
	if n < 0 {
		return ""
	}
	return C.GoStringN(&buf[0], n)
}

// Generate runs a prompt through the model with an optional GBNF grammar
// constraining the output (this is how JSON-schema-constrained extraction
// is enforced). Returns the generated text.
func (c *Context) Generate(prompt string, grammar string, maxTokens int) (string, int, error) {
	toks, err := c.m.tokenize(prompt, true)
	if err != nil {
		return "", 0, err
	}

	// sampler chain: greedy (temperature 0 per bench determinism policy),
	// with grammar constraint first if provided
	sparams := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(sparams)
	if grammar != "" {
		cg := C.CString(grammar)
		croot := C.CString("root")
		gs := C.llama_sampler_init_grammar(c.m.vocab, cg, croot)
		C.free(unsafe.Pointer(cg))
		C.free(unsafe.Pointer(croot))
		if gs == nil {
			return "", 0, fmt.Errorf("grammar failed to parse")
		}
		C.llama_sampler_chain_add(chain, gs)
	}
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_greedy())
	defer C.llama_sampler_free(chain)

	// prefill in n_batch-sized chunks: a single oversized llama_decode call
	// trips GGML_ASSERT(n_tokens_all <= n_batch) and SIGABRTs the process
	// (found by the overflow-tier bench workload).
	const chunk = 2048 // llama_context_default_params n_batch
	if len(toks) > c.nCtx-8 {
		return "", 0, fmt.Errorf("prompt (%d tokens) exceeds context %d", len(toks), c.nCtx)
	}
	for off := 0; off < len(toks); off += chunk {
		end := off + chunk
		if end > len(toks) {
			end = len(toks)
		}
		batch := C.llama_batch_get_one(&toks[off], C.int32_t(end-off))
		if C.llama_decode(c.ctx, batch) != 0 {
			return "", 0, fmt.Errorf("decode failed on prefill chunk at %d", off)
		}
	}

	var out strings.Builder
	generated := 0
	for generated < maxTokens {
		tok := C.llama_sampler_sample(chain, c.ctx, -1)
		if C.llama_vocab_is_eog(c.m.vocab, tok) {
			break
		}
		out.WriteString(c.m.detokenize(tok))
		generated++
		tokCopy := tok
		batch := C.llama_batch_get_one(&tokCopy, 1)
		if C.llama_decode(c.ctx, batch) != 0 {
			return out.String(), generated, fmt.Errorf("decode failed at token %d", generated)
		}
	}
	return out.String(), generated, nil
}

// Embed runs the prompt through an embeddings-mode context and returns the
// mean-pooled, L2-normalized sentence vector. The model must be an embedding
// model (e.g. nomic-embed) and the context created with NewEmbedContext.
func (c *Context) Embed(text string) ([]float32, error) {
	toks, err := c.m.tokenize(text, true)
	if err != nil {
		return nil, err
	}
	if len(toks) > c.nCtx-4 {
		toks = toks[:c.nCtx-4]
	}
	// clear KV between embeds: each call is an independent sequence
	C.llama_memory_clear(C.llama_get_memory(c.ctx), C.bool(true))
	batch := C.llama_batch_get_one(&toks[0], C.int32_t(len(toks)))
	if C.llama_decode(c.ctx, batch) != 0 {
		return nil, fmt.Errorf("embed decode failed")
	}
	nEmbd := int(C.llama_model_n_embd(c.m.model))
	ptr := C.llama_get_embeddings_seq(c.ctx, 0)
	if ptr == nil {
		ptr = C.llama_get_embeddings(c.ctx)
	}
	if ptr == nil {
		return nil, fmt.Errorf("no embeddings returned")
	}
	raw := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
	out := make([]float32, nEmbd)
	var norm float64
	for i, v := range raw {
		out[i] = v
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		inv := float32(1.0 / (math.Sqrt(norm)))
		for i := range out {
			out[i] *= inv
		}
	}
	return out, nil
}

// NewEmbedContext creates a context in embeddings mode (mean pooling).
func (m *Model) NewEmbedContext(nCtx, nThreads int) (*Context, error) {
	ctx := C.llama_init_from_model(m.model, C.spike_embed_ctx_params(C.int(nCtx), C.int(nThreads)))
	if ctx == nil {
		return nil, fmt.Errorf("failed to create embed context")
	}
	return &Context{ctx: ctx, m: m, nCtx: nCtx}, nil
}

func (c *Context) Free()  { C.llama_free(c.ctx) }
func (m *Model) Free()    { C.llama_model_free(m.model); C.llama_backend_free() }
