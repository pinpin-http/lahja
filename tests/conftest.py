import pytest

from lahja.config import corpus_dirs


@pytest.fixture
def real_split():
    """The split built by `make data`, when it exists (skipped in CI)."""
    import pandas as pd

    path = corpus_dirs("open")["interim"] / "split.parquet"
    if not path.exists():
        pytest.skip("run `make data` first")
    return pd.read_parquet(path)
