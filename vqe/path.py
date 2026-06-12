from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from iqml.models.params import count_parameters
from iqml.vqe.ansatz import VQEConfig, he_vqe_energy
from iqml.vqe.chemistry import PauliHamiltonian

Params = dict[str, Any]


@dataclass(frozen=True)
class PathVQEModelConfig:
    feature_dim: int
    hidden_dim: int = 16
    head_hidden_dim: int = 32
    residual_scale: float = 0.05
    base_scale: float = 0.05


def build_path_features(
    spacings: list[float] | tuple[float, ...] | jnp.ndarray,
    hf_energies: list[float] | tuple[float, ...] | jnp.ndarray | None = None,
) -> jnp.ndarray:
    spacing = jnp.asarray(spacings, dtype=jnp.float32)
    if spacing.ndim != 1 or spacing.shape[0] == 0:
        raise ValueError("spacings must be a non-empty 1D sequence")
    if hf_energies is None:
        hf = jnp.zeros_like(spacing)
    else:
        hf = jnp.asarray(hf_energies, dtype=jnp.float32)
        if hf.shape != spacing.shape:
            raise ValueError("hf_energies must have the same shape as spacings")

    delta = jnp.concatenate([jnp.zeros((1,), dtype=jnp.float32), spacing[1:] - spacing[:-1]])
    center = jnp.mean(spacing)
    scale = jnp.maximum(jnp.std(spacing), jnp.asarray(1e-6, dtype=jnp.float32))
    spacing_norm = (spacing - center) / scale
    index = jnp.arange(spacing.shape[0], dtype=jnp.float32) / float(max(spacing.shape[0] - 1, 1))
    hf_center = jnp.mean(hf)
    hf_scale = jnp.maximum(jnp.std(hf), jnp.asarray(1e-6, dtype=jnp.float32))
    hf_norm = (hf - hf_center) / hf_scale
    return jnp.stack([spacing, delta, spacing_norm, hf_norm, index], axis=-1)


def init_path_vqe_params(
    key: jax.Array,
    vqe_config: VQEConfig,
    model_config: PathVQEModelConfig,
    method: str,
) -> Params:
    method_key = method.lower()
    if method_key == "independent_path":
        return {
            "theta": model_config.base_scale
            * jax.random.normal(
                key,
                (1, vqe_config.depth, vqe_config.num_qubits, 2),
                dtype=jnp.float32,
            )
        }
    if method_key == "mlp_path":
        keys = jax.random.split(key, 4)
        return {
            "theta_base": model_config.base_scale
            * jax.random.normal(
                keys[0],
                (1, vqe_config.depth, vqe_config.num_qubits, 2),
                dtype=jnp.float32,
            ),
            "mlp": {
                "W1": _glorot(keys[1], (model_config.feature_dim, model_config.head_hidden_dim)),
                "b1": jnp.zeros((model_config.head_hidden_dim,), dtype=jnp.float32),
                "W2": _glorot(
                    keys[2],
                    (model_config.head_hidden_dim, vqe_config.depth * vqe_config.num_qubits * 2),
                ),
                "b2": jnp.zeros((vqe_config.depth * vqe_config.num_qubits * 2,), dtype=jnp.float32),
            },
        }
    if method_key == "lstm_path":
        keys = jax.random.split(key, 4)
        return {
            "theta_base": model_config.base_scale
            * jax.random.normal(
                keys[0],
                (1, vqe_config.depth, vqe_config.num_qubits, 2),
                dtype=jnp.float32,
            ),
            "lstm": {
                "W": _glorot(keys[1], (model_config.feature_dim + model_config.hidden_dim, 4 * model_config.hidden_dim)),
                "b": jnp.zeros((4 * model_config.hidden_dim,), dtype=jnp.float32),
            },
            "head": {
                "W": _glorot(keys[2], (model_config.hidden_dim, vqe_config.depth * vqe_config.num_qubits * 2)),
                "b": jnp.zeros((vqe_config.depth * vqe_config.num_qubits * 2,), dtype=jnp.float32),
            },
        }
    raise ValueError("method must be independent_path, mlp_path, or lstm_path")


def path_vqe_theta(
    params: Params,
    path_features: jnp.ndarray,
    vqe_config: VQEConfig,
    model_config: PathVQEModelConfig,
    method: str,
) -> jnp.ndarray:
    _validate_path_features(path_features, model_config)
    method_key = method.lower()
    num_points = path_features.shape[0]
    if method_key == "independent_path":
        base = params["theta"]
        if base.shape[0] == 1:
            base = jnp.repeat(base, num_points, axis=0)
        if base.shape[0] != num_points:
            raise ValueError("independent_path theta must have one row or match the number of path points")
        return jnp.clip(base, -jnp.pi, jnp.pi)
    if method_key == "mlp_path":
        hidden = jnp.tanh(path_features @ params["mlp"]["W1"] + params["mlp"]["b1"])
        raw = hidden @ params["mlp"]["W2"] + params["mlp"]["b2"]
    elif method_key == "lstm_path":
        hidden = _lstm_forward(params["lstm"], path_features, model_config.hidden_dim)
        raw = hidden @ params["head"]["W"] + params["head"]["b"]
    else:
        raise ValueError("method must be independent_path, mlp_path, or lstm_path")
    raw = raw.reshape(num_points, vqe_config.depth, vqe_config.num_qubits, 2)
    layer_deltas = model_config.residual_scale * jnp.tanh(raw)
    theta = params["theta_base"] + jnp.cumsum(layer_deltas, axis=1)
    return jnp.clip(theta, -jnp.pi, jnp.pi)


def path_vqe_mean_energy(
    params: Params,
    path_features: jnp.ndarray,
    hamiltonians: tuple[PauliHamiltonian, ...],
    vqe_config: VQEConfig,
    model_config: PathVQEModelConfig,
    method: str,
    hamiltonian_operators: tuple[Any, ...] | None = None,
) -> jnp.ndarray:
    theta = path_vqe_theta(params, path_features, vqe_config, model_config, method)
    values = []
    for index, hamiltonian in enumerate(hamiltonians):
        operator = None if hamiltonian_operators is None else hamiltonian_operators[index]
        values.append(he_vqe_energy(theta[index], hamiltonian, vqe_config, operator))
    return jnp.mean(jnp.asarray(values))


def path_vqe_energies(
    params: Params,
    path_features: jnp.ndarray,
    hamiltonians: tuple[PauliHamiltonian, ...],
    vqe_config: VQEConfig,
    model_config: PathVQEModelConfig,
    method: str,
    hamiltonian_operators: tuple[Any, ...] | None = None,
) -> jnp.ndarray:
    theta = path_vqe_theta(params, path_features, vqe_config, model_config, method)
    values = []
    for index, hamiltonian in enumerate(hamiltonians):
        operator = None if hamiltonian_operators is None else hamiltonian_operators[index]
        values.append(he_vqe_energy(theta[index], hamiltonian, vqe_config, operator))
    return jnp.asarray(values)


def theta_diagnostics(theta: jnp.ndarray) -> dict[str, float]:
    """Summarize layerwise smoothness/correlation of generated VQE parameters."""
    if theta.ndim == 3:
        flat = theta
    elif theta.ndim == 4:
        flat = theta.reshape(theta.shape[0], theta.shape[1], -1)
    else:
        raise ValueError("theta must have shape (num_points, depth, params) or (num_points, depth, num_qubits, 2)")
    if theta.shape[1] < 2:
        return {
            "theta_adjacent_cosine_mean": 0.0,
            "theta_adjacent_cosine_min": 0.0,
            "theta_delta_norm_mean": 0.0,
            "theta_delta_norm_max": 0.0,
            "theta_abs_mean": float(jnp.mean(jnp.abs(theta))),
            "theta_abs_max": float(jnp.max(jnp.abs(theta))),
        }

    left = flat[:, :-1, :]
    right = flat[:, 1:, :]
    dot = jnp.sum(left * right, axis=-1)
    denom = jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1)
    cosine = dot / jnp.maximum(denom, 1e-12)
    deltas = right - left
    delta_norm = jnp.linalg.norm(deltas, axis=-1)
    return {
        "theta_adjacent_cosine_mean": float(jnp.mean(cosine)),
        "theta_adjacent_cosine_min": float(jnp.min(cosine)),
        "theta_delta_norm_mean": float(jnp.mean(delta_norm)),
        "theta_delta_norm_max": float(jnp.max(delta_norm)),
        "theta_abs_mean": float(jnp.mean(jnp.abs(theta))),
        "theta_abs_max": float(jnp.max(jnp.abs(theta))),
    }


def path_vqe_parameter_count(params: Params) -> float:
    return float(count_parameters(params))


def _validate_path_features(path_features: jnp.ndarray, model_config: PathVQEModelConfig) -> None:
    if path_features.ndim != 2:
        raise ValueError("path_features must have shape (num_points, feature_dim)")
    if path_features.shape[-1] != model_config.feature_dim:
        raise ValueError(
            f"path_features feature dim {path_features.shape[-1]} must equal {model_config.feature_dim}"
        )


def _lstm_forward(params: Params, x: jnp.ndarray, hidden_dim: int) -> jnp.ndarray:
    h0 = jnp.zeros((hidden_dim,), dtype=jnp.float32)
    c0 = jnp.zeros((hidden_dim,), dtype=jnp.float32)

    def step(carry: tuple[jnp.ndarray, jnp.ndarray], x_t: jnp.ndarray):
        h, c = carry
        joined = jnp.concatenate([x_t, h], axis=-1)
        gates = joined @ params["W"] + params["b"]
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        next_c = f * c + i * g
        next_h = o * jnp.tanh(next_c)
        return (next_h, next_c), next_h

    (_, _), hidden_sequence = jax.lax.scan(step, (h0, c0), x)
    return hidden_sequence


def _glorot(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)
