VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X github.com/CarriedWorldUniverse/bridle/internal/version.Version=$(VERSION)

.PHONY: build test vet stubfunnel version clean

build:
	go build ./...

stubfunnel:
	go build -ldflags '$(LDFLAGS)' -o bin/stubfunnel ./stubfunnel

test:
	go test -race ./...

vet:
	go vet ./...

version:
	@echo $(VERSION)

clean:
	rm -rf bin/

# --- ctxmap in-process models (optional; build with -tags ctxmap_llama) ---
LLAMA_CPP ?= $(HOME)/src/llama.cpp

## vendor-llama: stage llama.cpp shared libs + headers for the ctxmap extractor/embedder
vendor-llama:
	mkdir -p vendor-llama/llama.cpp/lib vendor-llama/llama.cpp/include vendor-llama/llama.cpp/ggml/include
	cd $(LLAMA_CPP) && cmake -B build -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j $$(nproc) --target llama
	cp -P $(LLAMA_CPP)/build/bin/libllama.so* $(LLAMA_CPP)/build/bin/libggml*.so* vendor-llama/llama.cpp/lib/
	cp $(LLAMA_CPP)/include/*.h vendor-llama/llama.cpp/include/
	cp $(LLAMA_CPP)/ggml/include/*.h vendor-llama/llama.cpp/ggml/include/
