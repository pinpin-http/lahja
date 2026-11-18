import numpy as np
import pandas as pd

from lahja.config import CLASSES
from lahja.data import assign_splits, spread_sample


def synthetic(n_groups=300, per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for k in range(per_group):  # a parallel corpus: every group has all classes
            rows.append({"group_id": f"g{g}", "label": CLASSES[k % len(CLASSES)], "text_norm": f"s{g} {k}"})
    df = pd.DataFrame(rows)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def test_groups_never_cross_splits():
    df = synthetic()
    df["split"] = assign_splits(df)
    per_group = df.groupby("group_id").split.nunique()
    assert (per_group == 1).all()


def test_split_proportions():
    df = synthetic()
    share = assign_splits(df).value_counts(normalize=True)
    assert abs(share["test"] - 0.1) < 0.02
    assert abs(share["val"] - 0.1) < 0.02
    assert abs(share["train"] - 0.8) < 0.03


def test_split_is_deterministic():
    df = synthetic()
    assert assign_splits(df).equals(assign_splits(df))


def test_spread_sample_takes_groups_in_turn():
    df = pd.DataFrame({"group_id": ["big"] * 100 + ["a", "b", "c"], "text_norm": range(103)})
    picked = spread_sample(df, 6, np.random.default_rng(0))
    assert len(picked) == 6
    assert set(picked.group_id) == {"big", "a", "b", "c"}
    assert (picked.group_id == "big").sum() == 3


def test_real_split_has_no_shared_source_sentence(real_split):
    """No source sentence, TV show or dictionary headword in both training and test (or validation)."""
    c = real_split[real_split.set == "corpus"]
    groups = {s: set(c.loc[c.split == s, "group_id"]) for s in ("train", "val", "test")}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["val"]
    assert not groups["val"] & groups["test"]


def test_real_split_has_no_shared_text(real_split):
    c = real_split[real_split.set == "corpus"]
    train = set(c.loc[c.split == "train", "text_norm"])
    assert not train & set(c.loc[c.split == "test", "text_norm"])
    assert not train & set(c.loc[c.split == "val", "text_norm"])
    ood = real_split[real_split.split == "ood"]
    assert not set(ood.text_norm) & set(c.text_norm)


def test_real_split_respects_scope(real_split):
    assert real_split.n_words.between(3, 60).all()
    assert set(real_split.label) <= set(CLASSES)
