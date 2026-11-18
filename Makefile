# Lahja. `make all` goes from the downloads to the published model.
# Everything runs on a laptop CPU except `finetune` and `seeds`, which want a GPU.
export OMP_NUM_THREADS ?= 2
export MPLBACKEND = Agg

CORPUS ?= open
RUN ?= uv run python -m lahja
LAHJA = $(RUN) --corpus $(CORPUS)
MODELS_FT = camelbert-da marbert
SEEDS ?= 14 15

.PHONY: install data ingest normalize label split baseline finetune seeds export evaluate report publish \
        all serve test lint space clean

install:
	uv sync --frozen --group train --group dev

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

# folder ready to push to a Hugging Face Docker Space
space:
	rm -rf dist/space && mkdir -p dist/space/models
	git archive HEAD pyproject.toml uv.lock README.md src | tar -x -C dist/space
	cp space/Dockerfile dist/space/Dockerfile && cp space/README.md dist/space/README.md
	cp -r models/release dist/space/models/release
	@echo "dist/space is ready"

clean:
	rm -rf data/interim models results/$(CORPUS)
