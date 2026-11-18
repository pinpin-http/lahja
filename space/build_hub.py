"""Assemble dist/hub, the Hugging Face model repo: quantised model, tokenizer, model card
(English as README.md, French alongside) and the evaluation report with its figures."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hub"
LINKS = {
    "en": "**Demo:** [lahja.soheib.fr](https://lahja.soheib.fr) · **API:** "
    "[Space](https://huggingface.co/spaces/pinpin-http/lahja) · by [Soheib](https://soheib.fr)",
    "fr": "**Démo :** [lahja.soheib.fr](https://lahja.soheib.fr) · **API :** "
    "[Space](https://huggingface.co/spaces/pinpin-http/lahja) · par [Soheib](https://soheib.fr)",
}


def card(path: Path, lang: str) -> str:
    lines = path.read_text().splitlines()
    # after the title and the language link
    lines[3:3] = ["", LINKS[lang]]
    return "\n".join(lines).replace("(model_card.md)", "(README.md)") + "\n"


def main() -> int:
    release = ROOT / "models" / "release"
    if not (release / "model.int8.onnx").exists():
        print("no released model: run `make publish` first", file=sys.stderr)
        return 1
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "figures").mkdir(parents=True)
    for name in ("model.int8.onnx", "tokenizer.json", "lahja.json", "SHA256SUMS"):
        shutil.copy2(release / name, OUT / name)
    for name in ("evaluation.md", "evaluation.fr.md"):
        shutil.copy2(ROOT / "docs" / name, OUT / name)
    for f in (ROOT / "docs" / "figures").glob("*.png"):
        shutil.copy2(f, OUT / "figures" / f.name)
    header = (ROOT / "space" / "model_header.md").read_text()
    (OUT / "README.md").write_text(header + "\n" + card(ROOT / "docs" / "model_card.md", "en"))
    (OUT / "model_card.fr.md").write_text(card(ROOT / "docs" / "model_card.fr.md", "fr"))
    print(f"{OUT} is ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
