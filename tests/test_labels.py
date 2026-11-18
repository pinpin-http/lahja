from lahja import labels


def test_madar_cities():
    assert labels.madar_label("RIY") == ("gulf", "SA")
    assert labels.madar_label("Doha") == ("gulf", "QA")
    assert labels.madar_label("Muscat") == ("gulf", "OM")
    assert labels.madar_label("BEI") == ("levant", "LB")
    assert labels.madar_label("Jerusalem") == ("levant", "PS")
    assert labels.madar_label("ALX") == ("egypt", "EG")
    assert labels.madar_label("Tunis") == ("maghreb", "TN")
    assert labels.madar_label("BEN") == ("maghreb", "LY")
    assert labels.madar_label("MSA") == ("msa", None)


def test_out_of_scope_cities_dropped():
    for city in ("Baghdad", "BAS", "Mosul", "SAN", "Khartoum"):
        assert labels.madar_label(city)[0] is None
    assert labels.madar_label("Atlantis") == (None, None)


def test_countries():
    assert labels.country_label("ae") == "gulf"
    assert labels.country_label("PL") == "levant"
    assert labels.country_label("IQ") is None
    assert labels.country_label("YE") is None
    assert labels.country_label("SD") is None


def test_flores_and_sada():
    assert labels.FLORES["arb_Arab"][0] == "msa"
    assert labels.FLORES["ars_Arab"][0] == "gulf"
    assert "acm_Arab" not in labels.FLORES  # Iraqi: out of v1
    assert labels.SADA["Hijazi"][0] == "gulf"
    assert "ModernStandardArabic" not in labels.SADA  # speaker-level label, not used
    assert "Yemeni" not in labels.SADA


def test_every_mapping_uses_known_classes():
    from lahja.config import CLASSES

    for table in (labels.FLORES, labels.SADA, labels.LAHGA):
        assert {c for c, _ in table.values()} <= set(CLASSES)
    assert {c for c in labels.COUNTRY_CLASS.values() if c} <= set(CLASSES)
