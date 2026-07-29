from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_flat_walk_split_is_subject_disjoint_and_complete() -> None:
    split = json.loads(
        (PROJECT / "configs/flat_walk_split.json").read_text(encoding="utf-8")
    )
    train = split["train"]
    validation = split["validation"]
    assert len(train) == 229
    assert len(validation) == 52
    assert {row["subject"] for row in train}.isdisjoint(
        {row["subject"] for row in validation}
    )
    expected_categories = set(split["flat_categories"])
    assert {row["category"] for row in train} == expected_categories
    assert {row["category"] for row in validation} == expected_categories
    assert len({row["dataset_basename"] for row in train + validation}) == 281
