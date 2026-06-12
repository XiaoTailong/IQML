from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


FASHION_MNIST_FILENAMES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

FASHION_MNIST_URLS = {
    key: (
        f"https://storage.googleapis.com/tensorflow/tf-keras-datasets/{filename}",
        f"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{filename}",
        f"https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/{filename}",
    )
    for key, filename in FASHION_MNIST_FILENAMES.items()
}


@dataclass(frozen=True)
class VisionDataset:
    train_x: jnp.ndarray
    train_y: jnp.ndarray
    val_x: jnp.ndarray
    val_y: jnp.ndarray
    test_x: jnp.ndarray
    test_y: jnp.ndarray
    metadata: dict[str, Any]


def build_fashion_mnist_patch_dataset(config: dict[str, Any]) -> VisionDataset:
    data_dir = Path(str(config.get("data_dir", "data/fashion_mnist"))).expanduser()
    download = bool(config.get("download", True))
    patch_size = int(config.get("patch_size", 7))
    patch_order = str(config.get("patch_order", "snake"))
    add_position_features = bool(config.get("add_position_features", True))
    val_size = int(config.get("val_size", 10000))
    train_limit = _optional_int(config.get("train_limit"))
    val_limit = _optional_int(config.get("val_limit"))
    test_limit = _optional_int(config.get("test_limit"))

    if 28 % patch_size != 0:
        raise ValueError("patch_size must divide 28")
    paths = ensure_fashion_mnist_files(data_dir, download=download)
    train_images = _read_idx_images(paths["train_images"])
    train_labels = _read_idx_labels(paths["train_labels"])
    test_images = _read_idx_images(paths["test_images"])
    test_labels = _read_idx_labels(paths["test_labels"])

    if val_size <= 0 or val_size >= train_images.shape[0]:
        raise ValueError("val_size must be in (0, number of training images)")
    val_images = train_images[-val_size:]
    val_labels = train_labels[-val_size:]
    train_images = train_images[:-val_size]
    train_labels = train_labels[:-val_size]

    train_images, train_labels = _limit(train_images, train_labels, train_limit)
    val_images, val_labels = _limit(val_images, val_labels, val_limit)
    test_images, test_labels = _limit(test_images, test_labels, test_limit)

    train_x = _patchify(train_images, patch_size, patch_order, add_position_features)
    val_x = _patchify(val_images, patch_size, patch_order, add_position_features)
    test_x = _patchify(test_images, patch_size, patch_order, add_position_features)
    metadata = {
        "dataset": "FashionMNIST",
        "patch_size": patch_size,
        "patch_order": patch_order,
        "add_position_features": add_position_features,
        "seq_len": train_x.shape[1],
        "input_dim": train_x.shape[2],
        "num_classes": 10,
        "train_samples": train_x.shape[0],
        "val_samples": val_x.shape[0],
        "test_samples": test_x.shape[0],
    }
    return VisionDataset(
        train_x=jnp.asarray(train_x, dtype=jnp.float32),
        train_y=jnp.asarray(train_labels, dtype=jnp.int32),
        val_x=jnp.asarray(val_x, dtype=jnp.float32),
        val_y=jnp.asarray(val_labels, dtype=jnp.int32),
        test_x=jnp.asarray(test_x, dtype=jnp.float32),
        test_y=jnp.asarray(test_labels, dtype=jnp.int32),
        metadata=metadata,
    )


def ensure_fashion_mnist_files(data_dir: Path, *, download: bool) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: data_dir / filename for key, filename in FASHION_MNIST_FILENAMES.items()}
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing and not download:
        missing_files = ", ".join(str(paths[key]) for key in missing)
        raise FileNotFoundError(f"missing Fashion-MNIST files: {missing_files}")
    for key in missing:
        _download_with_fallback(FASHION_MNIST_URLS[key], paths[key])
    return paths


def _download_with_fallback(urls: tuple[str, ...], path: Path) -> None:
    errors = []
    for url in urls:
        try:
            print(f"downloading {url} -> {path}", flush=True)
            urllib.request.urlretrieve(url, path)
            return
        except Exception as exc:  # pragma: no cover - depends on network state
            if path.exists():
                path.unlink()
            errors.append(f"{url}: {exc}")
    message = "\n".join(errors)
    raise RuntimeError(f"failed to download {path.name} from all mirrors:\n{message}")


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"invalid IDX image file {path}: magic={magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"invalid IDX label file {path}: magic={magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count).astype(np.int32)


def _patchify(
    images: np.ndarray,
    patch_size: int,
    patch_order: str,
    add_position_features: bool,
) -> np.ndarray:
    count, height, width = images.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = images.reshape(count, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(0, 1, 3, 2, 4)
    patches = patches.reshape(count, grid_h * grid_w, patch_size * patch_size)
    order = _patch_order_indices(grid_h, grid_w, patch_order)
    patches = patches[:, order, :]
    if not add_position_features:
        return patches
    positions = _position_features(grid_h, grid_w)[order]
    tiled_positions = np.broadcast_to(positions[None, :, :], (count, positions.shape[0], positions.shape[1]))
    return np.concatenate([patches, tiled_positions.astype(np.float32)], axis=-1)


def _patch_order_indices(grid_h: int, grid_w: int, patch_order: str) -> np.ndarray:
    key = patch_order.lower().replace("-", "_")
    if key == "raster":
        return np.arange(grid_h * grid_w, dtype=np.int32)
    if key == "snake":
        rows = []
        for row in range(grid_h):
            cols = range(grid_w) if row % 2 == 0 else range(grid_w - 1, -1, -1)
            rows.extend(row * grid_w + col for col in cols)
        return np.asarray(rows, dtype=np.int32)
    raise ValueError("patch_order must be 'snake' or 'raster'")


def _position_features(grid_h: int, grid_w: int) -> np.ndarray:
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    row = rows.reshape(-1).astype(np.float32)
    col = cols.reshape(-1).astype(np.float32)
    row_norm = 2.0 * row / max(grid_h - 1, 1) - 1.0
    col_norm = 2.0 * col / max(grid_w - 1, 1) - 1.0
    radius = np.sqrt(row_norm**2 + col_norm**2)
    return np.stack([row_norm, col_norm, radius], axis=-1).astype(np.float32)


def _limit(images: np.ndarray, labels: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    if limit is None:
        return images, labels
    return images[:limit], labels[:limit]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
