"""
prepare_cuad_dataset.py
-----------------------
Loads the CUAD (Contract Understanding Atticus Dataset), extracts
(clause_text, clause_type) pairs from the SQuAD-format annotations, and
saves contract-level train/val/test splits to data/processed/.

The split is done BY CONTRACT (not by example) to prevent data leakage —
all examples from one source contract land in exactly one split.

Data source:
    The canonical CUAD release ships as a single ~40 MB `CUADv1.json`
    (SQuAD format, 510 contracts x 41 clause categories) inside the
    Atticus Project's `data.zip`. The old HuggingFace loader script no
    longer works with datasets>=5.0, so we read the JSON directly and, if
    it isn't already on disk, download + extract it automatically.

Usage:
    python training/prepare_cuad_dataset.py
    python training/prepare_cuad_dataset.py --local-path data/raw/CUADv1.json
    python training/prepare_cuad_dataset.py --output-dir data/processed
"""

import argparse
import io
import re
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_JSON_NAME = "CUADv1.json"
DEFAULT_RAW_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# CUAD loading
# ---------------------------------------------------------------------------

def _ensure_cuad_json(local_path: str | None) -> Path:
    """Return a path to CUADv1.json, downloading + extracting it if needed."""
    if local_path:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"--local-path given but not found: {p}")
        return p

    target = DEFAULT_RAW_DIR / CUAD_JSON_NAME
    if target.exists():
        print(f"    Using cached CUAD file: {target}")
        return target

    DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading CUAD data.zip (~18 MB) from {CUAD_ZIP_URL} ...")
    with urllib.request.urlopen(CUAD_ZIP_URL) as resp:  # noqa: S310 (trusted URL)
        blob = resp.read()
    print(f"    Extracting {CUAD_JSON_NAME} ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open(CUAD_JSON_NAME) as src, open(target, "wb") as dst:
            dst.write(src.read())
    print(f"    [OK] wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def _load_cuad_records(local_path: str | None) -> list[dict]:
    """Parse CUADv1.json (SQuAD format) into flat annotation records.

    Each record: {title, question, answers: {text: [...]}}. Questions with
    no answer span (is_impossible / empty answers) are kept here and filtered
    downstream, so the caller can report how many were skipped.
    """
    json_path = _ensure_cuad_json(local_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records: list[dict] = []
    for article in raw.get("data", []):
        title = article.get("title", "unknown")
        for para in article.get("paragraphs", []):
            for qa in para.get("qas", []):
                records.append(
                    {
                        "title": title,
                        "question": qa.get("question", ""),
                        "answers": {
                            "text": [a["text"] for a in qa.get("answers", [])],
                        },
                    }
                )
    print(f"    [OK] parsed {len(records)} annotation rows "
          f"from {len(raw.get('data', []))} contracts")
    if not records:
        raise RuntimeError(f"No records parsed from {json_path}")
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_clause_type_from_question(question: str) -> str:
    """Parse the clause category from a CUAD question string.

    CUAD questions follow patterns like:
      'Highlight the parts (if any) related to "Governing Law".'
      'Highlight the parts (if any) related to "Non-Compete" ...'

    Falls back to cleaning up the whole question if quotes are missing.
    """
    # Try extracting from double-quotes first
    match = re.search(r'"([^"]+)"', question)
    if match:
        return match.group(1).strip()
    # Try single quotes
    match = re.search(r"'([^']+)'", question)
    if match:
        return match.group(1).strip()
    # Fallback: strip common prefixes
    cleaned = question.strip().rstrip(".").rstrip("?")
    cleaned = re.sub(
        r"^Highlight the parts?\s*\(?if any\)?\s*(?:related to|that discuss|regarding)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip().strip("\"'")


def normalize_clause_type(raw_type: str) -> str:
    """Convert a raw clause type name to UPPER_SNAKE_CASE.

    'Governing Law'        -> 'GOVERNING_LAW'
    'Non-Compete'          -> 'NON_COMPETE'
    'Ip Ownership Assignment' -> 'IP_OWNERSHIP_ASSIGNMENT'
    """
    # Replace hyphens, slashes, spaces with underscores; uppercase
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", raw_type).upper().strip("_")
    return normalized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_dataset(
    local_path: str | None = None,
    output_dir: str = "data/processed",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CUAD, extract pairs, split by contract, and save CSVs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    print("=" * 60)
    print("STEP 1: Loading CUAD dataset")
    print("=" * 60)
    raw_records = _load_cuad_records(local_path)

    # ---- Extract clause-text pairs ----
    print("\n" + "=" * 60)
    print("STEP 2: Extracting (clause_text, clause_type) pairs")
    print("=" * 60)

    records: list[dict] = []
    skipped_empty = 0
    question_types_seen: set[str] = set()

    for row in raw_records:
        title = row.get("title", "unknown")
        question = row.get("question", "")
        answers = row.get("answers", {})
        answer_texts = answers.get("text", [])

        raw_type = extract_clause_type_from_question(question)
        clause_type = normalize_clause_type(raw_type)
        question_types_seen.add(clause_type)

        if not answer_texts or all(not t.strip() for t in answer_texts):
            skipped_empty += 1
            continue

        for answer_text in answer_texts:
            text = answer_text.strip()
            if not text:
                continue
            records.append(
                {
                    "clause_text": text,
                    "clause_type": clause_type,
                    "source_contract": title,
                }
            )

    df = pd.DataFrame(records)
    print(f"\n  Total raw examples: {len(df)}")
    print(f"  Skipped (no answer text): {skipped_empty}")
    print(f"  Unique clause types: {len(question_types_seen)}")

    # Deduplicate (same text + type + contract)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clause_text", "clause_type", "source_contract"])
    print(f"  After dedup: {len(df)} (removed {before_dedup - len(df)} duplicates)")
    print(f"  Unique contracts: {df['source_contract'].nunique()}")

    # ---- Contract-level split ----
    print("\n" + "=" * 60)
    print("STEP 3: Splitting by source contract (no leakage)")
    print("=" * 60)

    # .tolist() -> plain Python list -> object ndarray, so sklearn can index
    # it. (pandas 3.0 returns a pyarrow-backed array from .unique() that
    # train_test_split can't index with an integer position array.)
    contracts = np.array(df["source_contract"].unique().tolist(), dtype=object)
    np.random.seed(seed)

    # First split: train vs (val+test)
    train_contracts, valtest_contracts = train_test_split(
        contracts,
        test_size=(1 - train_ratio),
        random_state=seed,
    )
    # Second split: val vs test (50/50 of the remaining)
    val_contracts, test_contracts = train_test_split(
        valtest_contracts,
        test_size=0.5,
        random_state=seed,
    )

    train_df = df[df["source_contract"].isin(train_contracts)].reset_index(drop=True)
    val_df = df[df["source_contract"].isin(val_contracts)].reset_index(drop=True)
    test_df = df[df["source_contract"].isin(test_contracts)].reset_index(drop=True)

    print(f"  Contracts — train: {len(train_contracts)}, val: {len(val_contracts)}, test: {len(test_contracts)}")
    print(f"  Examples  — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ---- Leakage check ----
    train_set = set(train_contracts)
    val_set = set(val_contracts)
    test_set = set(test_contracts)
    assert train_set.isdisjoint(val_set), "LEAKAGE: train ∩ val is not empty!"
    assert train_set.isdisjoint(test_set), "LEAKAGE: train ∩ test is not empty!"
    assert val_set.isdisjoint(test_set), "LEAKAGE: val ∩ test is not empty!"
    print("  [OK] Leakage check passed: no contract appears in multiple splits")

    # ---- Class distribution ----
    print("\n" + "=" * 60)
    print("STEP 4: Class distribution")
    print("=" * 60)

    type_counts = df["clause_type"].value_counts()
    dist_df = pd.DataFrame(
        {
            "clause_type": type_counts.index,
            "total": type_counts.values,
            "train": [
                len(train_df[train_df["clause_type"] == t]) for t in type_counts.index
            ],
            "val": [
                len(val_df[val_df["clause_type"] == t]) for t in type_counts.index
            ],
            "test": [
                len(test_df[test_df["clause_type"] == t]) for t in type_counts.index
            ],
        }
    )
    print(dist_df.to_string(index=False))

    # ---- Save ----
    print("\n" + "=" * 60)
    print("STEP 5: Saving to disk")
    print("=" * 60)

    train_path = output_path / "train.csv"
    val_path = output_path / "val.csv"
    test_path = output_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  Saved {train_path} ({len(train_df)} rows)")
    print(f"  Saved {val_path} ({len(val_df)} rows)")
    print(f"  Saved {test_path} ({len(test_df)} rows)")
    print("\nDone! Ready for Step 2 (training).")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare CUAD dataset for clause classification training."
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to a local CUAD JSON file (skips HuggingFace download).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save train/val/test CSVs.",
    )
    args = parser.parse_args()
    prepare_dataset(local_path=args.local_path, output_dir=args.output_dir)
