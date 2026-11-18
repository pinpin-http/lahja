FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    LAHJA_THREADS=2 \
    LAHJA_MODEL_DIR=/app/models/release

# the model lives in its own Hugging Face repo, pinned to a revision and checked against SHA256SUMS
ARG MODEL_REPO=pinpin-http/lahja
ARG MODEL_REVISION=a6ba3c5f93be0aa0e5a6b195f220ccdc98e5fd4c

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev && chown -R user /app

RUN mkdir -p models/release && cd models/release \
 && for f in model.int8.onnx tokenizer.json lahja.json SHA256SUMS; do \
      python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
        "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/$f" "$f"; \
    done \
 && sha256sum -c SHA256SUMS && chown -R user /app/models

USER user
EXPOSE 7860
CMD ["/app/.venv/bin/uvicorn", "lahja.api:app", "--host", "0.0.0.0", "--port", "7860"]
