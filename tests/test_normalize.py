import pytest

from lahja.normalize import normalize, word_count


@pytest.mark.parametrize("raw", ["أحمد", "إحمد", "آحمد", "ٱحمد"])
def test_alif_variants_unified(raw):
    assert normalize(raw) == "احمد"


def test_diacritics_and_tatweel_removed():
    assert normalize("مَدْرَسَةٌ") == "مدرسه"
    assert normalize("جمـــيل") == "جميل"
    assert normalize("الرَّحْمٰن") == "الرحمن"  # shadda and dagger alif


def test_ta_marbuta_and_alif_maqsura_folded():
    assert normalize("مدرسة") == normalize("مدرسه") == "مدرسه"
    assert normalize("على") == normalize("علي") == "علي"


def test_persian_letters_mapped():
    assert normalize("کتاب") == "كتاب"
    assert normalize("فی") == "في"


def test_dialect_letters_kept():
    # g and v letters carry dialect information
    assert normalize("ڨال") == "ڨال"
    assert normalize("گال") == "گال"
    assert normalize("ڤيديو") == "ڤيديو"


def test_punctuation_digits_latin_emoji_removed():
    assert normalize("شو هاد؟! 123 ok 😂 ،؛") == "شو هاد"
    assert normalize("٣ تفاحات") == "تفاحات"  # Arabic-Indic digits too


def test_urls_and_mentions_removed():
    assert normalize("@user شوف هاد https://t.co/abc www.x.com") == "شوف هاد"


def test_letter_runs_shortened():
    assert normalize("هههههههه") == "هه"
    assert normalize("وااااو") == "وااو"


def test_presentation_forms_folded():
    assert normalize("ﻻ") == "لا"


def test_whitespace_and_empty():
    assert normalize("  كيف \n\t الحال  ") == "كيف الحال"
    assert normalize("") == ""
    assert normalize("hello 123 !!") == ""
    assert word_count("") == 0
    assert word_count("كيف الحال") == 2


def test_idempotent():
    s = "إزيك يا جماعة؟ عاملين إيه النهارده... 😂"
    assert normalize(normalize(s)) == normalize(s)


def test_training_and_serving_share_the_function():
    """The API and the data pipeline import the very same normalize()."""
    from lahja import data, inference
    from lahja import normalize as module

    assert data.normalize is module.normalize
    assert inference.normalize is module.normalize
