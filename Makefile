# Lahja. `make all` goes from the downloads to the published model.
# Everything runs on a laptop CPU except `finetune` and `seeds`, which want a GPU.
# `make install` takes the PyTorch build for NVIDIA (CUDA 13.2); `make install TRAIN=train-rocm` the AMD one (ROCm 7.2).
export OMP_NUM_THREADS ?= 2
export MPLBACKEND = Agg

CORPUS ?= open
RUN ?= uv run python -m lahja
LAHJA = $(RUN) --corpus $(CORPUS)
MODELS_FT = camelbert-da marbert
SEEDS ?= 14 15
TRAIN ?= train

.PHONY: install data ingest normalize label split baseline finetune seeds export evaluate report publish \
        all serve test lint hub space deploy-space site deploy-site clean

install:
	uv sync --frozen --group $(TRAIN) --group dev

data:
	$(LAHJA) data

ingest normalize label split baseline export evaluate report publish:
	$(LAHJA) $@

finetune:
	for m in $(MODELS_FT); do $(LAHJA) finetune --model $$m || exit 1; done

seeds:
	for s in $(SEEDS); do for m in $(MODELS_FT); do $(LAHJA) finetune --model $$m --seed $$s || exit 1; done; done

all: data baseline finetune seeds export evaluate report publish

serve:
	uv run uvicorn lahja.api:app --host 127.0.0.1 --port 8000

test:
	uv run pytest

lint:
	uv run ruff check src tests

HF_USER ?= pinpin-http
HF = uvx --from huggingface_hub hf

# Hugging Face model repo: quantised model, tokenizer, model card, evaluation report
hub:
	python3 -I space/build_hub.py
	$(HF) repos create $(HF_USER)/lahja --repo-type model --exist-ok
	$(HF) upload $(HF_USER)/lahja dist/hub . --repo-type model

# Hugging Face Docker Space serving the API; it downloads the model from the model repo
space:
	rm -rf dist/space && mkdir -p dist/space
	git archive HEAD pyproject.toml uv.lock README.md src | tar -x -C dist/space
	cp space/Dockerfile dist/space/Dockerfile && cp space/README.md dist/space/README.md
	@echo "dist/space is ready"

deploy-space: space
	$(HF) repos create $(HF_USER)/lahja --repo-type space --space-sdk docker --exist-ok
	$(HF) upload $(HF_USER)/lahja dist/space . --repo-type space

# public page on Cloudflare: the same web app, which downloads the model from the Hub and runs it
# in the browser with ONNX Runtime Web (self-hosted next to the page)
MODEL_REVISION ?= a6ba3c5f93be0aa0e5a6b195f220ccdc98e5fd4c
MODEL_URL ?= https://huggingface.co/$(HF_USER)/lahja/resolve/$(MODEL_REVISION)
ORT_WEB ?= 1.31.1
site:
	rm -rf dist/site && mkdir -p dist/site/static/ort dist/cache
	test -f dist/cache/onnxruntime-web-$(ORT_WEB).tgz || \
	  (cd dist/cache && npm pack --silent onnxruntime-web@$(ORT_WEB) >/dev/null)
	tar -xzf dist/cache/onnxruntime-web-$(ORT_WEB).tgz -C dist/site/static/ort --strip-components=2 \
	  package/dist/ort.wasm.min.mjs package/dist/ort-wasm-simd-threaded.mjs package/dist/ort-wasm-simd-threaded.wasm
	cp src/lahja/web/*.js src/lahja/web/*.css src/lahja/web/*.json dist/site/static/
	sed 's|<meta name="lahja-model" content="">|<meta name="lahja-model" content="$(MODEL_URL)">|' \
	  src/lahja/web/index.html > dist/site/index.html
	cp site/_headers dist/site/_headers
	@echo "dist/site is ready"

deploy-site: site
	cd site && npx --yes wrangler@4 deploy

clean:
	rm -rf data/interim models results/$(CORPUS)
