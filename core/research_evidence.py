"""Reproducible research utilities for ablations and error analysis.

These helpers are intentionally separate from the deployed diagnosis path. They
do not load or replace the production checkpoint and do not change app outputs.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_augmentation: bool
    use_class_weights: bool
    use_ben_graham: bool
    fine_tune: bool


ABLATION_VARIANTS: Tuple[AblationVariant, ...] = (
    AblationVariant("full_system", True, True, True, True),
    AblationVariant("without_augmentation", False, True, True, True),
    AblationVariant("without_class_weights", True, False, True, True),
    AblationVariant("without_ben_graham", True, True, False, True),
    AblationVariant("frozen_efficientnet", True, True, True, False),
)


def set_reproducible_seed(seed: int = 42) -> None:
    """Set deterministic seeds before constructing datasets or models."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Calculate the required ablation metrics on the same held-out test set."""
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def run_ablation_suite(
    train_and_predict: Callable[[AblationVariant], Tuple[Sequence[int], Sequence[int]]],
    output_path: Optional[str | Path] = None,
    variants: Iterable[AblationVariant] = ABLATION_VARIANTS,
) -> pd.DataFrame:
    """Run controlled variants through a caller-owned notebook training function.

    ``train_and_predict`` must use identical data splits and return ``(y_true,
    y_pred)`` for the held-out test set. The production checkpoint is never
    touched by this function.
    """
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        y_true, y_pred = train_and_predict(variant)
        rows.append({"experiment": variant.name, **calculate_metrics(y_true, y_pred)})
    results = pd.DataFrame(rows)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
    return results


def build_error_analysis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Optional[np.ndarray] = None,
    quality_records: Optional[Sequence[Dict[str, Any]]] = None,
    explainability_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create ordinal error summaries and inspectable per-image error rows."""
    true_values = np.asarray(y_true, dtype=int)
    pred_values = np.asarray(y_pred, dtype=int)
    confidence = np.max(probabilities, axis=1) if probabilities is not None else np.full(len(true_values), np.nan)
    distance = np.abs(true_values - pred_values)
    matrix = confusion_matrix(true_values, pred_values).tolist()
    rows: List[Dict[str, Any]] = []

    for index, (true_label, predicted_label) in enumerate(zip(true_values, pred_values)):
        row: Dict[str, Any] = {
            "index": index,
            "true_label": int(true_label),
            "predicted_label": int(predicted_label),
            "correct": bool(true_label == predicted_label),
            "stage_distance": int(abs(true_label - predicted_label)),
            "confidence": None if np.isnan(confidence[index]) else float(confidence[index]),
        }
        if quality_records is not None and index < len(quality_records):
            row.update({f"quality_{key}": value for key, value in quality_records[index].items()})
        if explainability_records is not None and index < len(explainability_records):
            evidence = explainability_records[index]
            row["lesion_burden_pct"] = evidence.get("lesion_pct")
            row["attention_lesion_dice"] = evidence.get("overlap_analysis", {}).get("dice")
            row["evidence_consistency"] = evidence.get("consistency_status")
        rows.append(row)

    error_rows = pd.DataFrame(rows)
    wrong = error_rows[~error_rows["correct"]]
    high_confidence_errors = wrong[wrong["confidence"].fillna(-1) >= 0.80]
    low_confidence_cases = error_rows[error_rows["confidence"].fillna(2.0) < 0.70]
    return {
        "confusion_matrix": matrix,
        "exact_accuracy": float(np.mean(distance == 0)),
        "adjacent_error_rate": float(np.mean(distance == 1)),
        "distant_error_rate": float(np.mean(distance >= 2)),
        "mean_absolute_stage_error": float(np.mean(distance)),
        "mild_to_moderate_errors": int(np.sum((true_values == 1) & (pred_values == 2))),
        "moderate_to_severe_errors": int(np.sum((true_values == 2) & (pred_values == 3))),
        "high_confidence_error_count": int(len(high_confidence_errors)),
        "low_confidence_case_count": int(len(low_confidence_cases)),
        "error_rows": error_rows,
    }


def save_error_analysis(analysis: Dict[str, Any], output_dir: str | Path) -> None:
    """Save machine-readable error summaries without fabricating image evidence."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = analysis.get("error_rows", pd.DataFrame())
    rows.to_csv(destination / "error_cases.csv", index=False)
    serializable = {key: value for key, value in analysis.items() if key != "error_rows"}
    (destination / "error_analysis_summary.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )


def build_reproducibility_manifest(
    config: Dict[str, Any],
    output_path: str | Path,
    dataset_version: str,
    split_files: Sequence[str],
    checkpoint_files: Sequence[str],
    training_duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Write versions, hardware, source revision, splits, and run configuration."""
    manifest: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "dataset_version": dataset_version,
        "random_seed": config.get("random_seed", 42),
        "split_files": list(split_files),
        "checkpoint_files": list(checkpoint_files),
        "training_duration_seconds": training_duration_seconds,
        "configuration": config,
    }
    try:
        import tensorflow as tf

        manifest["tensorflow_version"] = tf.__version__
        manifest["gpu_devices"] = [device.name for device in tf.config.list_physical_devices("GPU")]
    except ImportError:
        manifest["tensorflow_version"] = None
        manifest["gpu_devices"] = []
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_split_files(split_files: Sequence[str], group_column: str = "group_id") -> Dict[str, Any]:
    """Verify split files exist and contain no shared patient/group identifiers."""
    frames = [pd.read_csv(path) for path in split_files]
    groups = [set(frame[group_column].astype(str)) for frame in frames]
    intersections = {
        f"split_{left}_split_{right}": len(groups[left] & groups[right])
        for left in range(len(groups))
        for right in range(left + 1, len(groups))
    }
    return {"group_column": group_column, "intersections": intersections, "zero_overlap": all(value == 0 for value in intersections.values())}