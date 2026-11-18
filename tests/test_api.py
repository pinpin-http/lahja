import numpy as np
import pytest
from fastapi.testclient import TestClient

from lahja import api
from lahja.config import CLASSES, MODELS
from lahja.inference import NoArabicText
from lahja.normalize import normalize


class FakeModel:
    """Stands in for the ONNX model: 'شو' means Levant, 'ايش' means Gulf."""

    classes = list(CLASSES)
    config = {"model": "fake"}

    def _probs(self, norm):
        p = np.full(len(CLASSES), 0.05)
        p[CLASSES.index("levant" if "شو" in norm else "gulf" if "ايش" in norm else "msa")] = 0.8
        return p / p.sum()

    def predict(self, text):
        norm = normalize(text)
        if not norm:
            raise NoArabicText(text)
        p = self._probs(norm)
        return {
            "label": CLASSES[int(p.argmax())],
            "probabilities": dict(zip(CLASSES, p.round(4).tolist(), strict=True)),
            "n_words": len(norm.split()),
        }

    def explain(self, text):
        res = self.predict(text)
        return res, [{"word": w, "weight": 0.5 if w in ("شو", "ايش") else 0.0} for w in text.split()]


@pytest.fixture
def client():
    api.app.dependency_overrides[api.get_model] = FakeModel
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_predict_returns_class_and_probabilities(client):
    r = client.post("/predict", json={"text": "شو عم تعمل هلق"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "levant"
    assert set(body["probabilities"]) == set(CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["n_words"] == 4
    assert body["in_scope"] is True
    assert body["words"] is None


def test_predict_with_explanation(client):
    body = client.post("/predict", json={"text": "ايش تبي الحين", "explain": True}).json()
    assert body["label"] == "gulf"
    assert [w["word"] for w in body["words"]] == ["ايش", "تبي", "الحين"]
    assert body["words"][0]["weight"] > 0


def test_short_text_flagged_out_of_scope(client):
    body = client.post("/predict", json={"text": "شو"}).json()
    assert body["in_scope"] is False


def test_non_arabic_text_rejected(client):
    assert client.post("/predict", json={"text": "hello world 123"}).status_code == 422
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": "ا" * 2001}).status_code == 422


def test_health_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/")
    assert r.status_code == 200 and "Lahja" in r.text
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.skipif(not (MODELS / "release" / "model.int8.onnx").exists(), reason="no released model")
def test_released_model_end_to_end():
    api.get_model.cache_clear()
    c = TestClient(api.app)
    body = c.post("/predict", json={"text": "إزيك يا جماعة عاملين إيه النهارده", "explain": True}).json()
    assert body["label"] in CLASSES
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert len(body["words"]) == 6


def test_cors_preflight(client):
    r = client.options(
        "/predict",
        headers={
            "Origin": "https://lahja.soheib.fr",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] in ("*", "https://lahja.soheib.fr")
