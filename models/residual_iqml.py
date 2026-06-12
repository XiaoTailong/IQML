from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp

from iqml.quantum.circuits import (
    QuantumCircuitConfig,
    feature_count,
    parameter_count,
    run_pqc_batch,
)

Params = dict[str, Any]


@dataclass(frozen=True)
class ResidualIQMLConfig:
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
    theta_source: str = "hidden"
    input_theta_scale: float = 1.0
    hidden_theta_scale: float = math.pi
    hidden_theta_eps: float = 1e-6
    hidden_theta_norm: str = "global_rms"


@dataclass(frozen=True)
class ResidualIQMLOutput:
    prediction: jnp.ndarray
    base_prediction: jnp.ndarray
    quantum_residual: jnp.ndarray
    quantum_features: jnp.ndarray
    normalized_quantum_features: jnp.ndarray
    theta: jnp.ndarray
    hidden_sequence: jnp.ndarray


def init_residual_iqml_params(key: jax.Array, config: ResidualIQMLConfig) -> Params:
    keys = jax.random.split(key, 8)
    quantum_config = _quantum_config(config, depth=config.quantum_depth)
    quantum_parameter_count = parameter_count(quantum_config)
    quantum_feature_count = feature_count(quantum_config, depth=config.quantum_depth)
    theta_source = _normalize_theta_source(config.theta_source)
    if theta_source == "hidden_direct" and config.hidden_dim != quantum_parameter_count:
        raise ValueError(
            "theta_source='hidden_direct' requires hidden_dim to equal the QNN "
            f"layer parameter count ({quantum_parameter_count}), got {config.hidden_dim}"
        )
    params = {
        "lstm": {
            "W": _glorot(keys[0], (config.input_dim + config.hidden_dim, 4 * config.hidden_dim)),
            "b": jnp.zeros((4 * config.hidden_dim,), dtype=jnp.float32),
        },
        "base_head": {
            "W": _glorot(keys[1], (config.hidden_dim, config.output_dim)),
            "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
        },
        "residual_head": {
            "hidden": {
                "W": _glorot(keys[3], (quantum_feature_count, config.fusion_hidden_dim)),
                "b": jnp.zeros((config.fusion_hidden_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": config.residual_init_scale
                * jax.random.normal(keys[4], (config.fusion_hidden_dim, config.output_dim), dtype=jnp.float32),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        },
    }
    if theta_source == "hidden":
        params["quantum_encoder"] = {
            "W": _glorot(keys[2], (config.hidden_dim, quantum_parameter_count)),
            "b": jnp.zeros((quantum_parameter_count,), dtype=jnp.float32),
        }
    elif theta_source not in {"hidden_direct", "input"}:
        raise ValueError("theta_source must be 'hidden', 'hidden_direct', or 'input'")
    return params


def residual_iqml_forward(
    params: Params,
    x: jnp.ndarray,
    config: ResidualIQMLConfig,
) -> ResidualIQMLOutput:
    if x.shape[1] != config.quantum_depth:
        raise ValueError(
            f"residual_iqml expects seq_len == quantum_depth, got {x.shape[1]} and {config.quantum_depth}"
        )
    hidden_sequence = _lstm_forward(params["lstm"], x, config.hidden_dim)
    last_hidden = hidden_sequence[:, -1, :]
    base_prediction = _linear(params["base_head"], last_hidden)
    theta = _quantum_theta(params, x, hidden_sequence, config)
    quantum_features = run_pqc_batch(theta, _quantum_config(config, depth=theta.shape[1]))
    normalized_features = _layer_norm(quantum_features)
    residual_hidden = jax.nn.relu(_linear(params["residual_head"]["hidden"], normalized_features))
    quantum_residual = _linear(params["residual_head"]["out"], residual_hidden)
    prediction = base_prediction + quantum_residual
    return ResidualIQMLOutput(
        prediction=prediction,
        base_prediction=base_prediction,
        quantum_residual=quantum_residual,
        quantum_features=quantum_features,
        normalized_quantum_features=normalized_features,
        theta=theta,
        hidden_sequence=hidden_sequence,
    )


def residual_iqml_predict_from_quantum_features(
    params: Params,
    base_prediction: jnp.ndarray,
    quantum_features: jnp.ndarray,
) -> jnp.ndarray:
    normalized_features = _layer_norm(quantum_features)
    residual_hidden = jax.nn.relu(_linear(params["residual_head"]["hidden"], normalized_features))
    quantum_residual = _linear(params["residual_head"]["out"], residual_hidden)
    return base_prediction + quantum_residual


def _quantum_theta(
    params: Params,
    x: jnp.ndarray,
    hidden_sequence: jnp.ndarray,
    config: ResidualIQMLConfig,
) -> jnp.ndarray:
    quantum_parameter_count = parameter_count(_quantum_config(config, depth=config.quantum_depth))
    theta_source = _normalize_theta_source(config.theta_source)
    if theta_source == "hidden":
        return jnp.pi * jnp.tanh(_linear(params["quantum_encoder"], hidden_sequence))
    if theta_source == "hidden_direct":
        if hidden_sequence.shape[-1] != quantum_parameter_count:
            raise ValueError(
                "theta_source='hidden_direct' requires hidden_sequence last dimension "
                f"to equal {quantum_parameter_count}, got {hidden_sequence.shape[-1]}"
            )
        return _normalize_hidden_direct_theta(hidden_sequence, config)
    if theta_source == "input":
        expanded = _repeat_features_to_dim(x, quantum_parameter_count)
        return jnp.pi * jnp.tanh(config.input_theta_scale * expanded)
    raise ValueError("theta_source must be 'hidden', 'hidden_direct', or 'input'")


def _repeat_features_to_dim(x: jnp.ndarray, output_dim: int) -> jnp.ndarray:
    repeats = (int(output_dim) + int(x.shape[-1]) - 1) // int(x.shape[-1])
    return jnp.tile(x, (1, 1, repeats))[:, :, :output_dim]


def _normalize_theta_source(theta_source: str) -> str:
    key = theta_source.lower().replace("-", "_")
    if key in {"hidden_direct", "lstm_hidden", "hidden_state"}:
        return "hidden_direct"
    return key


def _normalize_hidden_direct_theta(
    hidden_sequence: jnp.ndarray,
    config: ResidualIQMLConfig,
) -> jnp.ndarray:
    mode = config.hidden_theta_norm.lower().replace("-", "_")
    scale = jnp.asarray(config.hidden_theta_scale, dtype=hidden_sequence.dtype)
    eps = config.hidden_theta_eps
    if mode in {"global_rms", "sample_rms", "window_rms", "rms"}:
        normalized = _centered_rms_norm(hidden_sequence, eps=eps, axis=(1, 2))
    elif mode in {"layer_rms", "time_rms", "per_layer_rms", "per_time_rms"}:
        normalized = _centered_rms_norm(hidden_sequence, eps=eps, axis=-1)
    elif mode in {"layer_minmax_symmetric", "time_minmax_symmetric", "per_layer_minmax_symmetric"}:
        normalized = _minmax_symmetric_norm(hidden_sequence, eps=eps, axis=-1)
    elif mode in {"layer_minmax_positive", "time_minmax_positive", "per_layer_minmax_positive"}:
        normalized = _minmax_positive_norm(hidden_sequence, eps=eps, axis=-1)
    else:
        raise ValueError(
            "hidden_theta_norm must be one of: global_rms, layer_rms, "
            "layer_minmax_symmetric, or layer_minmax_positive"
        )
    return scale * normalized


def residual_iqml_mse_loss(
    params: Params,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: ResidualIQMLConfig,
) -> jnp.ndarray:
    output = residual_iqml_forward(params, x, config)
    return jnp.mean((output.prediction - y) ** 2)


def _quantum_config(
    config: ResidualIQMLConfig,
    depth: int | None = None,
) -> QuantumCircuitConfig:
    return QuantumCircuitConfig(
        num_qubits=config.num_qubits,
        circuit_type=config.circuit_type,
        observables=config.observables,
        entanglement=config.entanglement,
        readout_layers=_readout_layers(depth, config.readout_stride),
    )


def _readout_layers(depth: int | None, stride: int) -> tuple[int, ...] | None:
    if depth is None:
        return None
    if stride <= 0:
        return (int(depth),)
    layers = list(range(stride, int(depth) + 1, stride))
    if not layers or layers[-1] != int(depth):
        layers.append(int(depth))
    return tuple(layers)


def _lstm_forward(params: Params, x: jnp.ndarray, hidden_dim: int) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=params["W"].dtype)
    batch_size = x.shape[0]
    h0 = jnp.zeros((batch_size, hidden_dim), dtype=params["W"].dtype)
    c0 = jnp.zeros((batch_size, hidden_dim), dtype=params["W"].dtype)

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

    (_, _), hidden_time_major = jax.lax.scan(step, (h0, c0), jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(hidden_time_major, 0, 1)


def _linear(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["W"] + params["b"]


def _layer_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _centered_rms_norm(
    x: jnp.ndarray,
    *,
    eps: float = 1e-6,
    axis: tuple[int, ...] = (-1,),
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=axis, keepdims=True)
    centered = x - mean
    rms = jnp.sqrt(jnp.mean(centered**2, axis=axis, keepdims=True) + eps)
    return centered / rms


def _minmax_symmetric_norm(
    x: jnp.ndarray,
    *,
    eps: float = 1e-6,
    axis: int | tuple[int, ...] = -1,
) -> jnp.ndarray:
    xmin = jnp.min(x, axis=axis, keepdims=True)
    xmax = jnp.max(x, axis=axis, keepdims=True)
    span = jnp.maximum(xmax - xmin, eps)
    return 2.0 * (x - xmin) / span - 1.0


def _minmax_positive_norm(
    x: jnp.ndarray,
    *,
    eps: float = 1e-6,
    axis: int | tuple[int, ...] = -1,
) -> jnp.ndarray:
    xmin = jnp.min(x, axis=axis, keepdims=True)
    xmax = jnp.max(x, axis=axis, keepdims=True)
    span = jnp.maximum(xmax - xmin, eps)
    return (x - xmin) / span


def _glorot(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)
