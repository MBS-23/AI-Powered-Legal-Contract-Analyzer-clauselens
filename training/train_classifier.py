"""
train_classifier.py
--------------------
Trains a lightweight, deployable clause-type classifier on the CUAD splits
produced by prepare_cuad_dataset.py.

Design decisions (why this, not Legal-BERT):
  - The target deployment is Streamlit Community Cloud with no GPU and tight
    memory. A TF-IDF + linear model loads in milliseconds, is a few MB on
    disk, and needs no torch. That makes it the *right* production choice
    here, not a compromise.
  - We evaluate against a DummyClassifier (most-frequent) baseline so the
    lift from the real model is quantified honestly, and we report macro-F1
    (not just accuracy) because CUAD is heavily class-imbalanced.

Pipeline:
  TfidfVectorizer(word 1-2 grams) -> {LogisticRegression | LinearSVC}
  We fit both, pick the one with the better *validation* macro-F1, then
  report its held-out *test* metrics and persist it with joblib.

The saved artifact (models/clause_classifier.joblib) is a dict:
  { "pipeline", "labels", "threshold", "metrics", "sklearn_version" }
so the app can load it, predict, and apply a confidence threshold behind
the same ClauseMatch interface the Phase 1 keyword matcher used.

Usage:
    python training/train_classifier.py
    python training/train_classifier.py --data-dir data/processed --out models
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import joblib


def _load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.dropna(subset=["clause_text", "clause_type"], inplace=True)
        df["clause_text"] = df["clause_text"].astype(str)
        if df.empty:
            raise RuntimeError(f"{name} split is empty after cleaning")
    return train, val, test


def _build_pipeline(estimator) -> Pipeline:
    """TF-IDF features + a linear classifier.

    The vectorizer settings are tuned for legal prose: word 1-2 grams catch
    phrases like "governing law" / "hold harmless"; sublinear_tf dampens the
    effect of very long clauses; min_df=2 drops one-off OCR noise.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            ("clf", estimator),
        ]
    )


def _candidate_models() -> dict[str, Pipeline]:
    """Two linear candidates. Both expose predict_proba (LinearSVC via
    calibration) so the app can threshold on confidence."""
    logreg = LogisticRegression(
        max_iter=2000,
        C=10.0,
        class_weight="balanced",
    )
    # LinearSVC often edges out LogReg on sparse text; wrap it so we still
    # get calibrated probabilities for the confidence threshold.
    svc = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced"),
        method="sigmoid",
        cv=3,
    )
    return {
        "logreg": _build_pipeline(logreg),
        "linsvc_calibrated": _build_pipeline(svc),
    }


def train(data_dir: str = "data/processed", out_dir: str = "models") -> dict:
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("STEP 1: Loading splits")
    print("=" * 64)
    train_df, val_df, test_df = _load_splits(data_path)
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"  classes: {train_df['clause_type'].nunique()}")

    X_train, y_train = train_df["clause_text"], train_df["clause_type"]
    X_val, y_val = val_df["clause_text"], val_df["clause_type"]
    X_test, y_test = test_df["clause_text"], test_df["clause_type"]

    # Only train on classes that appear in train; drop val/test rows whose
    # label was never seen in training (can't be predicted anyway).
    train_labels = set(y_train.unique())
    val_mask = y_val.isin(train_labels)
    test_mask = y_test.isin(train_labels)
    X_val, y_val = X_val[val_mask], y_val[val_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    print("\n" + "=" * 64)
    print("STEP 2: Baseline (most-frequent DummyClassifier)")
    print("=" * 64)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
    dummy_f1 = f1_score(y_test, dummy.predict(X_test), average="macro", zero_division=0)
    print(f"  baseline test accuracy = {dummy_acc:.3f}")
    print(f"  baseline test macro-F1 = {dummy_f1:.3f}")

    print("\n" + "=" * 64)
    print("STEP 3: Fitting candidate models, selecting on validation macro-F1")
    print("=" * 64)
    best_name, best_pipe, best_val_f1 = None, None, -1.0
    for name, pipe in _candidate_models().items():
        print(f"  fitting {name} ...")
        pipe.fit(X_train, y_train)
        val_f1 = f1_score(y_val, pipe.predict(X_val), average="macro", zero_division=0)
        print(f"    {name} validation macro-F1 = {val_f1:.3f}")
        if val_f1 > best_val_f1:
            best_name, best_pipe, best_val_f1 = name, pipe, val_f1
    print(f"  --> selected: {best_name} (val macro-F1 {best_val_f1:.3f})")

    print("\n" + "=" * 64)
    print("STEP 4: Held-out test evaluation")
    print("=" * 64)
    y_pred = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"  test accuracy     = {test_acc:.3f}")
    print(f"  test macro-F1     = {test_macro_f1:.3f}")
    print(f"  test weighted-F1  = {test_weighted_f1:.3f}")
    print(f"  lift over baseline: "
          f"accuracy +{test_acc - dummy_acc:.3f}, macro-F1 +{test_macro_f1 - dummy_f1:.3f}")

    report = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )

    # Pick a confidence threshold: the app treats predictions below this as
    # "no confident match". We choose the value that maximizes accuracy on
    # confidently-predicted validation rows while still covering most of them.
    proba = best_pipe.predict_proba(X_val)
    max_proba = proba.max(axis=1)
    pred_val = best_pipe.classes_[proba.argmax(axis=1)]
    threshold = _choose_threshold(max_proba, pred_val, y_val.to_numpy())
    print(f"\n  chosen confidence threshold = {threshold:.2f}")

    metrics = {
        "baseline_accuracy": float(dummy_acc),
        "baseline_macro_f1": float(dummy_f1),
        "selected_model": best_name,
        "val_macro_f1": float(best_val_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "n_classes": int(len(best_pipe.classes_)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "confidence_threshold": float(threshold),
        "per_class_f1": {
            label: round(vals["f1-score"], 3)
            for label, vals in report.items()
            if label not in ("accuracy", "macro avg", "weighted avg")
        },
    }

    print("\n" + "=" * 64)
    print("STEP 5: Persisting model + metrics")
    print("=" * 64)
    artifact = {
        "pipeline": best_pipe,
        "labels": list(best_pipe.classes_),
        "threshold": float(threshold),
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    model_file = out_path / "clause_classifier.joblib"
    metrics_file = out_path / "metrics.json"
    joblib.dump(artifact, model_file, compress=3)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  saved model   -> {model_file} ({model_file.stat().st_size / 1e6:.1f} MB)")
    print(f"  saved metrics -> {metrics_file}")
    print("\nDone.")
    return metrics


def _choose_threshold(
    max_proba: np.ndarray, pred: np.ndarray, truth: np.ndarray
) -> float:
    """Grid-search a confidence cutoff that keeps precision high on the
    kept predictions without discarding too much coverage."""
    best_t, best_score = 0.15, -1.0
    for t in np.arange(0.10, 0.60, 0.05):
        keep = max_proba >= t
        if keep.sum() == 0:
            continue
        precision = (pred[keep] == truth[keep]).mean()
        coverage = keep.mean()
        # Balance precision and coverage; slight preference for precision.
        score = 0.7 * precision + 0.3 * coverage
        if score > best_score:
            best_score, best_t = score, float(t)
    return round(best_t, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the clause-type classifier.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    train(data_dir=args.data_dir, out_dir=args.out)
