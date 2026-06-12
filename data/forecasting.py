from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from iqml.data.windowing import DatasetSplits, WindowedDataset


@dataclass(frozen=True)
class ForecastingDataset:
    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset
    metadata: dict[str, Any]


def build_csv_forecasting_dataset(config: dict[str, Any]) -> ForecastingDataset:
    """Load a real-valued multivariate CSV and build chronological forecast windows."""
    path = Path(str(config["path"])).expanduser()
    feature_columns = list(config.get("feature_columns") or [])
    target_column = str(config.get("target_column", "OT"))
    seq_len = int(config.get("seq_len", config.get("window_size", 96)))
    pred_len = int(config.get("pred_len", config.get("horizon", 96)))
    train_ratio = float(config.get("train_ratio", 0.6))
    val_ratio = float(config.get("val_ratio", 0.2))
    max_rows = config.get("max_rows")

    if not feature_columns:
        raise ValueError("feature_columns must contain at least one numeric column")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if pred_len <= 0:
        raise ValueError("pred_len must be positive")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    rows = _read_numeric_columns(path, feature_columns, target_column, max_rows)
    features = rows["features"]
    target = rows["target"]
    total_rows = int(features.shape[0])
    if total_rows < seq_len + pred_len + 3:
        raise ValueError("CSV has too few rows for the requested seq_len and pred_len")

    train_end = int(total_rows * train_ratio)
    val_end = train_end + int(total_rows * val_ratio)
    if train_end <= seq_len + pred_len:
        raise ValueError("training split is too short for the requested windows")
    mean = jnp.mean(features[:train_end], axis=0, keepdims=True)
    std = jnp.maximum(jnp.std(features[:train_end], axis=0, keepdims=True), 1e-6)
    target_mean = jnp.mean(target[:train_end])
    target_std = jnp.maximum(jnp.std(target[:train_end]), 1e-6)

    scaled_features = (features - mean) / std
    scaled_target = (target - target_mean) / target_std

    train = _window_range(scaled_features, scaled_target, 0, train_end, seq_len, pred_len)
    val = _window_range(scaled_features, scaled_target, train_end - seq_len, val_end, seq_len, pred_len)
    test = _window_range(scaled_features, scaled_target, val_end - seq_len, total_rows, seq_len, pred_len)
    metadata = {
        "path": str(path),
        "rows": total_rows,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "input_dim": len(feature_columns),
        "output_dim": pred_len,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "train_rows": train_end,
        "val_rows": val_end - train_end,
        "test_rows": total_rows - val_end,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
    }
    return ForecastingDataset(train=train, val=val, test=test, metadata=metadata)


def _read_numeric_columns(
    path: Path,
    feature_columns: list[str],
    target_column: str,
    max_rows: object,
) -> dict[str, jnp.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"forecasting CSV not found: {path}")
    limit = int(max_rows) if max_rows is not None else None
    feature_values: list[list[float]] = []
    target_values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in [*feature_columns, target_column] if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV {path} is missing required columns: {missing}")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            feature_values.append([float(row[column]) for column in feature_columns])
            target_values.append(float(row[target_column]))
    if not feature_values:
        raise ValueError(f"CSV {path} did not contain numeric rows")
    return {
        "features": jnp.asarray(feature_values, dtype=jnp.float32),
        "target": jnp.asarray(target_values, dtype=jnp.float32),
    }


def _window_range(
    features: jnp.ndarray,
    target: jnp.ndarray,
    start: int,
    stop: int,
    seq_len: int,
    pred_len: int,
) -> WindowedDataset:
    first = max(0, int(start))
    last = int(stop) - seq_len - pred_len + 1
    if last <= first:
        raise ValueError(
            f"split is too short for seq_len={seq_len} and pred_len={pred_len}: "
            f"start={start}, stop={stop}"
        )
    xs = [features[index : index + seq_len] for index in range(first, last)]
    ys = [target[index + seq_len : index + seq_len + pred_len] for index in range(first, last)]
    return WindowedDataset(
        x=jnp.stack(xs).astype(jnp.float32),
        y=jnp.stack(ys).astype(jnp.float32),
    )
