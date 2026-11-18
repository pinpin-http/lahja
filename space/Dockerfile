FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    LAHJA_THREADS=2 \
    LAHJA_MODEL_DIR=/app/models/release

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev && chown -R user /app
COPY --chown=user models/release ./models/release

USER user
EXPOSE 7860
CMD ["/app/.venv/bin/uvicorn", "lahja.api:app", "--host", "0.0.0.0", "--port", "7860"]
