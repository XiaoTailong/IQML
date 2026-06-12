from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp


@dataclass(frozen=True)
class WindowedDataset:
    x: jnp.ndarray
    y: jnp.ndarray


@dataclass(frozen=True)
class DatasetSplits:
    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset


def make_supervised_windows(
    series: jnp.ndarray,
    window_size: int,
    horizon: int = 1,
    target_column: int = 0,
    input_columns: Sequence[int] | None = None,
) -> WindowedDataset:
    """Convert a sequence into sliding-window multi-step regression samples.

    For ``horizon == 1`` this matches the previous one-step behavior. For
    larger horizons, each target is the direct future trajectory
    ``series[t + window_size : t + window_size + horizon, target_column]``.
    The future axis is flattened into the output dimension.
    """
    if series.ndim == 1:
        series = series[:, None]
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    count = series.shape[0] - window_size - horizon + 1
    if count <= 0:
        raise ValueError("series is too short for the requested window")

    if input_columns is not None:
        series_x = series[:, jnp.asarray(input_columns)]
    else:
        series_x = series

    xs = [series_x[i : i + window_size] for i in range(count)]
    ys = [
        series[i + window_size : i + window_size + horizon, target_column].reshape(-1)
        for i in range(count)
    ]
    return WindowedDataset(
        x=jnp.stack(xs).astype(jnp.float32),
        y=jnp.asarray(ys, dtype=jnp.float32).reshape(count, horizon),
    )


def split_dataset(
    dataset: WindowedDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> DatasetSplits:
    """Split samples chronologically into train, validation, and test sets."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    n = dataset.x.shape[0]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return DatasetSplits(
        train=WindowedDataset(dataset.x[:train_end], dataset.y[:train_end]),
        val=WindowedDataset(dataset.x[train_end:val_end], dataset.y[train_end:val_end]),
        test=WindowedDataset(dataset.x[val_end:], dataset.y[val_end:]),
    )
