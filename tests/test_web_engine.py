"""The browser engine (src/lahja/web/engine.js) must normalise and tokenise exactly like the
Python package. Needs Node; the tokeniser half also needs the released model."""

import json
import shutil
import subprocess

import pytest

from lahja.config import MODELS, ROOT
from lahja.normalize import normalize

NODE = shutil.which("node")
ENGINE = ROOT / "src" / "lahja" / "web" / "engine.js"
TOKENIZER = MODELS / "release" / "tokenizer.json"
TEXTS = [
    "شو عم تعمل هلق؟",
    "ٱلسَّلَامُ عَلَيْكُمْ وَرَحْمَةُ ٱللَّهِ",
    "شوووووو هاد؟؟ https://t.co/abc @user_1 test@mail.com www.example.com",
    "سؤال عن الطائرة ئ ؤ إ أ آ",
    "ﻻ ﺃﻋﺮﻑ",  # presentation forms
    "ڤيديو ڨال ڭال گال پاپا چاي ژورنال ی ک ھ",
    "😂😂 ههههههه 123 abc ١٢٣",
    "واش راك لاباس\tعليك\nخويا",
    "",
    "   ",
    "hello world",
]

pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def run_node(script: str, payload) -> list:
    out = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_normalize_matches_python():
    script = f"""
import {{ normalize }} from {json.dumps(ENGINE.as_uri())};
let s = ""; for await (const c of process.stdin) s += c;
console.log(JSON.stringify(JSON.parse(s).map(normalize)));
"""
    assert run_node(script, TEXTS) == [normalize(t) for t in TEXTS]


@pytest.mark.skipif(not TOKENIZER.exists(), reason="no released model")
def test_tokenizer_matches_python():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    tok.no_padding()
    tok.enable_truncation(128)
    texts = [normalize(t) for t in TEXTS] + [" ".join(["كلمة"] * 200)]
    script = f"""
import fs from "node:fs";
import {{ WordPiece }} from {json.dumps(ENGINE.as_uri())};
const wp = new WordPiece(JSON.parse(fs.readFileSync({json.dumps(str(TOKENIZER))}, "utf8")), 128);
let s = ""; for await (const c of process.stdin) s += c;
console.log(JSON.stringify(JSON.parse(s).map((t) => wp.encode(t))));
"""
    assert run_node(script, texts) == [e.ids for e in tok.encode_batch(texts)]
