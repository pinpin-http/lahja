"""HTTP API and demo page.

POST /predict  {"text": "...", "explain": false}
  -> {"label": "gulf", "probabilities": {...}, "n_words": 7, "in_scope": true, "words": null}
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import MAX_WORDS, MIN_WORDS, MODELS
from .inference import DialectModel, NoArabicText

WEB = Path(__file__).parent / "web"


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    explain: bool = False


class Word(BaseModel):
    word: str
    weight: float


class PredictOut(BaseModel):
    label: str
    probabilities: dict[str, float]
    n_words: int
    in_scope: bool
    words: list[Word] | None = None


@lru_cache(maxsize=1)
def get_model() -> DialectModel:
    return DialectModel(os.environ.get("LAHJA_MODEL_DIR", MODELS / "release"))


app = FastAPI(title="Lahja", version=__version__, docs_url="/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health(model: Annotated[DialectModel, Depends(get_model)]) -> dict:
    return {"status": "ok", "model": model.config.get("model"), "classes": model.classes}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn, model: Annotated[DialectModel, Depends(get_model)]) -> PredictOut:
    try:
        if body.explain:
            res, words = model.explain(body.text)
        else:
            res, words = model.predict(body.text), None
    except NoArabicText:
        raise HTTPException(status_code=422, detail="no Arabic text") from None
    return PredictOut(**res, in_scope=MIN_WORDS <= res["n_words"] <= MAX_WORDS, words=words)
