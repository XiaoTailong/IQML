from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp

from iqml.models.iqml import _lstm_forward
from iqml.quantum.circuits import QuantumCircuitConfig, feature_count, observable_count

Params = dict[str, Any]


@dataclass(frozen=True)
class ClassicalProjectionConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_qubits: int
    circuit_type: str = "he"
    fusion_hidden_dim: int = 16
    entanglement: str = "linear"
    projection_activation: str = "tanh"
    projection_pool: str = "last"


@dataclass(frozen=True)
class ClassicalProjectionOutput:
    prediction: jnp.ndarray
    classical_prediction: jnp.ndarray
    projected_features: jnp.ndarray
    hidden_sequence: jnp.ndarray


@dataclass(frozen=True)
class ResidualClassicalProjectionConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_qubits: int = 8
    circuit_type: str = "he"
    fusion_hidden_dim: int = 24
    entanglement: str = "linear"
    observables: str = "z,zz"
    quantum_depth: int = 48
    readout_stride: int = 0
    residual_init_scale: float = 0.01
    projection_activation: str = "tanh"
    projection_pool: str = "last"


@dataclass(frozen=True)
class ResidualClassicalProjectionOutput:
    prediction: jnp.ndarray
    base_prediction: jnp.ndarray
    classical_residual: jnp.ndarray
    projected_features: jnp.ndarray
    normalized_projected_features: jnp.ndarray
    hidden_sequence: jnp.ndarray


def init_classical_projection_params(key: jax.Array, config: ClassicalProjectionConfig) -> Params:
    keys = jax.random.split(key, 6)
    feature_dim = int(config.num_qubits)
    return {
        "lstm": {
            "W": _glorot(keys[0], (config.input_dim + config.hidden_dim, 4 * config.hidden_dim)),
            "b": jnp.zeros((4 * config.hidden_dim,), dtype=jnp.float32),
        },
        "classical_head": {
            "W": _glorot(keys[1], (config.hidden_dim, config.output_dim)),
            "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
        },
        "classical_projector": {
            "W": _glorot(keys[2], (config.hidden_dim, feature_dim)),
            "b": jnp.zeros((feature_dim,), dtype=jnp.float32),
        },
        "fusion": {
            "hidden": {
                "W": _glorot(keys[3], (config.output_dim + feature_dim, config.fusion_hidden_dim)),
                "b": jnp.zeros((config.fusion_hidden_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": _glorot(keys[4], (config.fusion_hidden_dim, config.output_dim)),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        },
    }


def classical_projection_forward(
    params: Params,
    x: jnp.ndarray,
    config: ClassicalProjectionConfig,
) -> ClassicalProjectionOutput:
    hidden_sequence = _lstm_forward(params["lstm"], x, config.hidden_dim)
    last_hidden = hidden_sequence[:, -1, :]
    classical_prediction = _linear(params["classical_head"], last_hidden)
    projected_features = _project_hidden_sequence(
        params["classical_projector"],
        hidden_sequence,
        activation=config.projection_activation,
        pool=config.projection_pool,
    )
    prediction = classical_projection_predict_from_features(
        params,
        classical_prediction,
        projected_features,
    )
    return ClassicalProjectionOutput(
        prediction=prediction,
        classical_prediction=classical_prediction,
        projected_features=projected_features,
        hidden_sequence=hidden_sequence,
    )


def classical_projection_predict_from_features(
    params: Params,
    classical_prediction: jnp.ndarray,
    projected_features: jnp.ndarray,
) -> jnp.ndarray:
    fusion_input = jnp.concatenate([classical_prediction, projected_features], axis=-1)
    hidden = jax.nn.relu(_linear(params["fusion"]["hidden"], fusion_input))
    return _linear(params["fusion"]["out"], hidden)


def classical_projection_mse_loss(
    params: Params,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: ClassicalProjectionConfig,
) -> jnp.ndarray:
    output = classical_projection_forward(params, x, config)
    return jnp.mean((output.prediction - y) ** 2)


def init_residual_classical_projection_params(
    key: jax.Array,
    config: ResidualClassicalProjectionConfig,
) -> Params:
    keys = jax.random.split(key, 6)
    projected_feature_count = _projected_feature_count(config)
    return {
        "lstm": {
            "W": _glorot(keys[0], (config.input_dim + config.hidden_dim, 4 * config.hidden_dim)),
            "b": jnp.zeros((4 * config.hidden_dim,), dtype=jnp.float32),
        },
        "base_head": {
            "W": _glorot(keys[1], (config.hidden_dim, config.output_dim)),
            "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
        },
        "classical_projector": {
            "W": _glorot(keys[2], (config.hidden_dim, projected_feature_count)),
            "b": jnp.zeros((projected_feature_count,), dtype=jnp.float32),
        },
        "residual_head": {
            "hidden": {
                "W": _glorot(keys[3], (projected_feature_count, config.fusion_hidden_dim)),
                "b": jnp.zeros((config.fusion_hidden_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": config.residual_init_scale
                * jax.random.normal(keys[4], (config.fusion_hidden_dim, config.output_dim), dtype=jnp.float32),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        },
    }


def residual_classical_projection_forward(
    params: Params,
    x: jnp.ndarray,
    config: ResidualClassicalProjectionConfig,
) -> ResidualClassicalProjectionOutput:
    if x.shape[1] != config.quantum_depth:
        raise ValueError(
            "residual_classical_projection expects seq_len == quantum_depth, "
            f"got {x.shape[1]} and {config.quantum_depth}"
        )
    hidden_sequence = _lstm_forward(params["lstm"], x, config.hidden_dim)
    last_hidden = hidden_sequence[:, -1, :]
    base_prediction = _linear(params["base_head"], last_hidden)
    projected_features = _project_hidden_sequence(
        params["classical_projector"],
        hidden_sequence,
        activation=config.projection_activation,
        pool=config.projection_pool,
    )
    normalized_features = _layer_norm(projected_features)
    residual_hidden = jax.nn.relu(_linear(params["residual_head"]["hidden"], normalized_features))
    classical_residual = _linear(params["residual_head"]["out"], residual_hidden)
    prediction = base_prediction + classical_residual
    return ResidualClassicalProjectionOutput(
        prediction=prediction,
        base_prediction=base_prediction,
        classical_residual=classical_residual,
        projected_features=projected_features,
        normalized_projected_features=normalized_features,
        hidden_sequence=hidden_sequence,
    )


def residual_classical_projection_mse_loss(
    params: Params,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: ResidualClassicalProjectionConfig,
) -> jnp.ndarray:
    output = residual_classical_projection_forward(params, x, config)
    return jnp.mean((output.prediction - y) ** 2)


def _project_hidden_sequence(
    projector: Params,
    hidden_sequence: jnp.ndarray,
    *,
    activation: str = "tanh",
    pool: str = "last",
) -> jnp.ndarray:
    projected_sequence = _apply_activation(_linear(projector, hidden_sequence), activation)
    pool_key = pool.lower().replace("-", "_")
    if pool_key == "last":
        return projected_sequence[:, -1, :]
    if pool_key in {"mean", "average"}:
        return jnp.mean(projected_sequence, axis=1)
    if pool_key in {"mean_last", "last_mean"}:
        return 0.5 * (projected_sequence[:, -1, :] + jnp.mean(projected_sequence, axis=1))
    raise ValueError("projection_pool must be 'last', 'mean', or 'mean_last'")


def _apply_activation(x: jnp.ndarray, activation: str) -> jnp.ndarray:
    key = activation.lower().replace("-", "_")
    if key == "tanh":
        return jnp.tanh(x)
    if key == "gelu":
        return jax.nn.gelu(x)
    if key == "linear":
        return x
    raise ValueError("projection_activation must be 'tanh', 'gelu', or 'linear'")


def _projected_feature_count(config: ResidualClassicalProjectionConfig) -> int:
    quantum_config = QuantumCircuitConfig(
        num_qubits=config.num_qubits,
        circuit_type=config.circuit_type,
        observables=config.observables,
        entanglement=config.entanglement,
        readout_layers=_readout_layers(config.quantum_depth, config.readout_stride),
    )
    return feature_count(quantum_config, depth=config.quantum_depth)


def residual_projected_feature_count(config: ResidualClassicalProjectionConfig) -> int:
    return _projected_feature_count(config)


def residual_observable_count(config: ResidualClassicalProjectionConfig) -> int:
    return observable_count(
        QuantumCircuitConfig(
            num_qubits=config.num_qubits,
            circuit_type=config.circuit_type,
            observables=config.observables,
            entanglement=config.entanglement,
        )
    )


def _readout_layers(depth: int, stride: int) -> tuple[int, ...]:
    if stride <= 0:
        return (int(depth),)
    layers = list(range(int(stride), int(depth) + 1, int(stride)))
    if not layers or layers[-1] != int(depth):
        layers.append(int(depth))
    return tuple(layers)


def _linear(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["W"] + params["b"]


def _layer_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _glorot(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)
