"""Arabic text normalisation, shared by training, evaluation and the API.

The same function runs everywhere, so a sentence typed in the demo goes through exactly
the steps the training data went through.

Steps, in order:
1. NFKC, which folds Arabic presentation forms back to base letters;
2. URLs, @mentions and e-mail addresses removed;
3. diacritics (harakat, shadda, sukun, dagger alif, Quranic marks) and tatweel removed;
4. alif variants (أ إ آ ٱ) to bare alif, ta marbuta (ة) to ha (ه), alif maqsura (ى) to ya (ي),
   Persian kaf and ya to their Arabic forms;
5. anything that is not an Arabic letter becomes a space: punctuation, digits, Latin script,
   emoji. Punctuation and digits differ between sources (subtitles have none, news has
   plenty) and would let a model guess the source instead of the dialect;
6. letters repeated three times or more cut down to two (هههههه -> هه);
7. whitespace collapsed.
"""

from __future__ import annotations

import re
import unicodedata

VERSION = 1

_URL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+|@\w+)")
_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ی": "ي",  # Persian ya
        "ې": "ي",
        "ک": "ك",  # Persian kaf (keheh)
        "ڪ": "ك",
        "ھ": "ه",  # heh doachashmee
        "ۀ": "ه",
    }
)
# Arabic letters U+0621-U+064A plus extended letters that dialects use for sounds MSA lacks:
# پ چ ژ ڤ (v, Maghreb) ڨ (g, Algeria/Tunisia) ڭ (g, Morocco) گ (g, Gulf/Iraq).
_NOT_ARABIC = re.compile("[^ء-غف-يپچژڤڨڭگ ]+")
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _URL.sub(" ", t)
    t = _DIACRITICS.sub("", t)
    t = t.translate(_MAP)
    t = _SPACES.sub(" ", t)
    t = _NOT_ARABIC.sub(" ", t)
    t = _REPEAT.sub(r"\1\1", t)
    return _SPACES.sub(" ", t).strip()


def word_count(normalized: str) -> int:
    return len(normalized.split()) if normalized else 0
