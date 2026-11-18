"""Mapping from each source's own labels (city, language code, speaker dialect, country) to the
five v1 classes. Anything mapped to None is left out of v1: Iraq, Yemen, Sudan, Mauritania,
and labels that do not name a dialect."""

from __future__ import annotations

# ISO 3166 alpha-2 -> class
COUNTRY_CLASS = {
    "SA": "gulf", "AE": "gulf", "QA": "gulf", "KW": "gulf", "BH": "gulf", "OM": "gulf",
    "SY": "levant", "LB": "levant", "JO": "levant", "PS": "levant", "PL": "levant",
    "EG": "egypt",
    "MA": "maghreb", "DZ": "maghreb", "TN": "maghreb", "LY": "maghreb",
    "IQ": None, "YE": None, "SD": None, "MR": None,
}  # fmt: skip

# FLORES-200 language codes. acm (Mesopotamian) and acq (Ta'izzi-Adeni) are out of v1.
FLORES = {
    "arb_Arab": ("msa", None),
    "ars_Arab": ("gulf", "SA"),
    "apc_Arab": ("levant", None),
    "ajp_Arab": ("levant", None),
    "arz_Arab": ("egypt", "EG"),
    "ary_Arab": ("maghreb", "MA"),
    "aeb_Arab": ("maghreb", "TN"),
}

# SADA speaker dialects. Every Saudi regional variety is Gulf. The "ModernStandardArabic"
# label describes the speaker, not the segment, and many of those segments are dialectal,
# so it is not used.
SADA = {
    "Najdi": ("gulf", "SA"),
    "Hijazi": ("gulf", "SA"),
    "Khaliji": ("gulf", "SA"),
    "Shamali": ("gulf", "SA"),
    "Janubi": ("gulf", "SA"),
    "Egyptian": ("egypt", "EG"),
    "Levantine": ("levant", None),
    "Maghrebi": ("maghreb", None),
}

# MADAR cities (codes as in the corpus files, plus full names) -> (class, country)
MADAR_CITIES = {
    "RIY": ("Riyadh", "SA"), "JED": ("Jeddah", "SA"), "DOH": ("Doha", "QA"), "MUS": ("Muscat", "OM"),
    "DAM": ("Damascus", "SY"), "ALE": ("Aleppo", "SY"), "BEI": ("Beirut", "LB"), "AMM": ("Amman", "JO"),
    "SAL": ("Salt", "JO"), "JER": ("Jerusalem", "PS"),
    "CAI": ("Cairo", "EG"), "ALX": ("Alexandria", "EG"), "ASW": ("Aswan", "EG"),
    "RAB": ("Rabat", "MA"), "FES": ("Fes", "MA"), "ALG": ("Algiers", "DZ"), "TUN": ("Tunis", "TN"),
    "SFX": ("Sfax", "TN"), "TRI": ("Tripoli", "LY"), "BEN": ("Benghazi", "LY"),
    "BAG": ("Baghdad", "IQ"), "BAS": ("Basra", "IQ"), "MOS": ("Mosul", "IQ"),
    "SAN": ("Sanaa", "YE"), "KHA": ("Khartoum", "SD"),
}  # fmt: skip
_MADAR_BY_NAME = {name.lower(): code for code, (name, _) in MADAR_CITIES.items()}

# Lahga dialect slugs -> (class, country)
LAHGA = {
    "egyptian": ("egypt", "EG"), "cairene": ("egypt", "EG"), "saidi": ("egypt", "EG"),
    "alexandrian": ("egypt", "EG"),
    "levantine": ("levant", None), "palestinian": ("levant", "PS"), "syrian": ("levant", "SY"),
    "lebanese": ("levant", "LB"), "jordanian": ("levant", "JO"),
    "gulf": ("gulf", None), "kuwaiti": ("gulf", "KW"), "bahraini": ("gulf", "BH"), "qatari": ("gulf", "QA"),
    "emirati": ("gulf", "AE"), "omani": ("gulf", "OM"), "eastern-saudi": ("gulf", "SA"),
    "najdi": ("gulf", "SA"), "hejazi": ("gulf", "SA"),
    "moroccan": ("maghreb", "MA"), "algerian": ("maghreb", "DZ"), "tunisian": ("maghreb", "TN"),
    "libyan": ("maghreb", "LY"), "maghrebi": ("maghreb", None),
    "msa_definition": ("msa", None),  # headword definitions, written in MSA
}  # fmt: skip


def madar_label(city: str) -> tuple[str | None, str | None]:
    """City code or name (or 'MSA') -> (class, country)."""
    key = city.strip()
    if key.upper() == "MSA":
        return "msa", None
    code = key.upper() if key.upper() in MADAR_CITIES else _MADAR_BY_NAME.get(key.lower())
    if code is None:
        return None, None
    country = MADAR_CITIES[code][1]
    return COUNTRY_CLASS.get(country), country


def country_label(code: str) -> str | None:
    return COUNTRY_CLASS.get(code.strip().upper())
