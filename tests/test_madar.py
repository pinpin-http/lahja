from pathlib import Path

import pandas as pd

from lahja.ingest import find_madar_files, parse_madar

FIX = Path(__file__).parent / "fixtures" / "madar"


def test_parse_madar_fixture():
    df = parse_madar(find_madar_files(FIX))
    assert set(df.columns) >= {"source", "group_id", "variety", "text", "country"}
    assert set(df.variety) == {"RIY", "CAI", "MSA"}  # English rows dropped
    # parallel corpus: the same sentence id across cities is one group
    assert df.groupby("group_id").variety.nunique().max() == 3
    assert (df.source == "madar").all()


def test_city_from_file_name(tmp_path):
    pd.DataFrame({"sentID.BTEC": ["1", "2"], "sent": ["إزيك", "عامل إيه"]}).to_csv(
        tmp_path / "MADAR.corpus.Cairo.tsv", sep="\t", index=False
    )
    df = parse_madar(find_madar_files(tmp_path))
    assert list(df.variety) == ["Cairo", "Cairo"]
    assert list(df.country) == ["EG", "EG"]
