"""
Tests for Phase 2 dataset preparation.

Validates:
  - No train/test leakage (contract-level split integrity)
  - All splits are non-empty with expected columns
  - Class labels are consistent across splits
  - No empty clause text
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

EXPECTED_COLUMNS = {"clause_text", "clause_type", "source_contract"}


@pytest.fixture
def splits():
    """Load the three split CSVs; skip the suite if they don't exist yet."""
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        pytest.skip(
            "data/processed/{train,val,test}.csv not found — "
            "run `python training/prepare_cuad_dataset.py` first."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


# ---------- Schema ----------


def test_splits_have_expected_columns(splits):
    train_df, val_df, test_df = splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert EXPECTED_COLUMNS.issubset(set(df.columns)), (
            f"{name} is missing columns: {EXPECTED_COLUMNS - set(df.columns)}"
        )


def test_splits_are_non_empty(splits):
    train_df, val_df, test_df = splits
    assert len(train_df) > 0, "train split is empty"
    assert len(val_df) > 0, "val split is empty"
    assert len(test_df) > 0, "test split is empty"


# ---------- Leakage ----------


def test_no_contract_leakage_train_val(splits):
    """No source contract should appear in both train and val."""
    train_df, val_df, _ = splits
    train_contracts = set(train_df["source_contract"].unique())
    val_contracts = set(val_df["source_contract"].unique())
    overlap = train_contracts & val_contracts
    assert len(overlap) == 0, (
        f"LEAKAGE: {len(overlap)} contracts in both train and val: "
        f"{list(overlap)[:5]}..."
    )


def test_no_contract_leakage_train_test(splits):
    """No source contract should appear in both train and test."""
    train_df, _, test_df = splits
    train_contracts = set(train_df["source_contract"].unique())
    test_contracts = set(test_df["source_contract"].unique())
    overlap = train_contracts & test_contracts
    assert len(overlap) == 0, (
        f"LEAKAGE: {len(overlap)} contracts in both train and test: "
        f"{list(overlap)[:5]}..."
    )


def test_no_contract_leakage_val_test(splits):
    """No source contract should appear in both val and test."""
    _, val_df, test_df = splits
    val_contracts = set(val_df["source_contract"].unique())
    test_contracts = set(test_df["source_contract"].unique())
    overlap = val_contracts & test_contracts
    assert len(overlap) == 0, (
        f"LEAKAGE: {len(overlap)} contracts in both val and test: "
        f"{list(overlap)[:5]}..."
    )


# ---------- Data quality ----------


def test_no_empty_clause_text(splits):
    """Every row should have non-empty clause_text."""
    train_df, val_df, test_df = splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        empty = df["clause_text"].isna() | (df["clause_text"].str.strip() == "")
        assert empty.sum() == 0, f"{name} has {empty.sum()} empty clause_text rows"


def test_no_empty_clause_type(splits):
    """Every row should have a clause_type label."""
    train_df, val_df, test_df = splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        empty = df["clause_type"].isna() | (df["clause_type"].str.strip() == "")
        assert empty.sum() == 0, f"{name} has {empty.sum()} empty clause_type rows"


def test_clause_types_are_uppercase_snake_case(splits):
    """All clause types should be UPPER_SNAKE_CASE (our normalization convention)."""
    import re

    train_df, val_df, test_df = splits
    all_types = set()
    for df in [train_df, val_df, test_df]:
        all_types.update(df["clause_type"].unique())

    pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
    bad = [t for t in all_types if not pattern.match(t)]
    assert len(bad) == 0, f"Clause types not in UPPER_SNAKE_CASE: {bad}"


def test_train_has_most_data(splits):
    """Train split should be the largest (roughly 80%)."""
    train_df, val_df, test_df = splits
    total = len(train_df) + len(val_df) + len(test_df)
    train_ratio = len(train_df) / total
    assert train_ratio > 0.65, (
        f"Train split is only {train_ratio:.1%} of total — expected ~80%"
    )
