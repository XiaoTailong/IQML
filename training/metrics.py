from __future__ import annotations

import jax.numpy as jnp


def regression_metrics(prediction: jnp.ndarray, target: jnp.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = jnp.mean(error**2)
    mae = jnp.mean(jnp.abs(error))
    rmse = jnp.sqrt(mse)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
    }
