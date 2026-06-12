from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp

from iqml.quantum.circuits import QuantumCircuitConfig, feature_count, parameter_count, run_pqc_batch

Params = dict[str, Any]


@dataclass(frozen=True)
class VisionLSTMConfig:
    input_dim: int
    hidden_dim: int
    num_classes: int = 10
    head_hidden_dim: int = 64


@dataclass(frozen=True)
class VisionIQMLConfig:
    input_dim: int
    hidden_dim: int
    num_classes: int = 10
    num_qubits: int = 8
    circuit_type: str = "he"
    fusion_hidden_dim: int = 64
    entanglement: str = "linear"
    observables: str = "z,zz"
    quantum_depth: int = 16
    readout_stride: int = 0
    fusion_mode: str = "logit_residual"
    residual_init_scale: float = 0.01
    theta_source: str = "hidden_direct"
    hidden_theta_scale: float = math.pi
    hidden_theta_eps: float = 1e-6
    hidden_theta_norm: str = "layer_rms"


@dataclass(frozen=True)
class VisionOutput:
    logits: jnp.ndarray
    hidden_sequence: jnp.ndarray


@dataclass(frozen=True)
class VisionIQMLOutput:
    logits: jnp.ndarray
    classical_logits: jnp.ndarray
    quantum_logits: jnp.ndarray
    quantum_features: jnp.ndarray
    normalized_quantum_features: jnp.ndarray
    theta: jnp.ndarray
    hidden_sequence: jnp.ndarray


def init_vision_lstm_params(key: jax.Array, config: VisionLSTMConfig) -> Params:
    keys = jax.random.split(key, 4)
    return {
        "lstm": {
            "W": _glorot(keys[0], (config.input_dim + config.hidden_dim, 4 * config.hidden_dim)),
            "b": jnp.zeros((4 * config.hidden_dim,), dtype=jnp.float32),
        },
        "head": {
            "hidden": {
                "W": _glorot(keys[1], (config.hidden_dim, config.head_hidden_dim)),
                "b": jnp.zeros((config.head_hidden_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": _glorot(keys[2], (config.head_hidden_dim, config.num_classes)),
                "b": jnp.zeros((config.num_classes,), dtype=jnp.float32),
            },
        },
    }


def init_vision_iqml_params(key: jax.Array, config: VisionIQMLConfig) -> Params:
    keys = jax.random.split(key, 8)
    quantum_config = _quantum_config(config, depth=config.quantum_depth)
    quantum_parameter_count = parameter_count(quantum_config)
    quantum_feature_count = feature_count(quantum_config, depth=config.quantum_depth)
    if _normalize_theta_source(config.theta_source) == "hidden_direct" and config.hidden_dim != quantum_parameter_count:
        raise ValueError(
            "theta_source='hidden_direct' requires hidden_dim to equal the QNN "
            f"layer parameter count ({quantum_parameter_count}), got {config.hidden_dim}"
        )
    params: Params = {
        "lstm": {
            "W": _glorot(keys[0], (config.input_dim + config.hidden_dim, 4 * config.hidden_dim)),
            "b": jnp.zeros((4 * config.hidden_dim,), dtype=jnp.float32),
        }
    }
    fusion_mode = _normalize_fusion_mode(config.fusion_mode)
    if fusion_mode == "logit_residual":
        params.update(
            {
                "classical_head": {
                    "W": _glorot(keys[1], (config.hidden_dim, config.num_classes)),
                    "b": jnp.zeros((config.num_classes,), dtype=jnp.float32),
                },
                "quantum_head": {
                    "hidden": {
                        "W": _glorot(keys[2], (quantum_feature_count, config.fusion_hidden_dim)),
                        "b": jnp.zeros((config.fusion_hidden_dim,), dtype=jnp.float32),
                    },
                    "out": {
                        "W": config.residual_init_scale
                        * jax.random.normal(keys[3], (config.fusion_hidden_dim, config.num_classes), dtype=jnp.float32),
                        "b": jnp.zeros((config.num_classes,), dtype=jnp.float32),
                    },
                },
            }
        )
    elif fusion_mode == "feature_concat":
        params["fusion_head"] = {
            "hidden": {
                "W": _glorot(keys[2], (config.hidden_dim + quantum_feature_count, config.fusion_hidden_dim)),
                "b": jnp.zeros((config.fusion_hidden_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": _glorot(keys[3], (config.fusion_hidden_dim, config.num_classes)),
                "b": jnp.zeros((config.num_classes,), dtype=jnp.float32),
            },
        }
    else:
        raise ValueError("fusion_mode must be 'logit_residual' or 'feature_concat'")
    if _normalize_theta_source(config.theta_source) == "hidden":
        params["quantum_encoder"] = {
            "W": _glorot(keys[4], (config.hidden_dim, quantum_parameter_count)),
            "b": jnp.zeros((quantum_parameter_count,), dtype=jnp.float32),
        }
    return params


def vision_lstm_forward(params: Params, x: jnp.ndarray, config: VisionLSTMConfig) -> VisionOutput:
    hidden_sequence = _lstm_forward(params["lstm"], x, config.hidden_dim)
    last_hidden = hidden_sequence[:, -1, :]
    head_hidden = jax.nn.relu(_linear(params["head"]["hidden"], last_hidden))
    logits = _linear(params["head"]["out"], head_hidden)
    return VisionOutput(logits=logits, hidden_sequence=hidden_sequence)


def vision_iqml_forward(params: Params, x: jnp.ndarray, config: VisionIQMLConfig) -> VisionIQMLOutput:
    if x.shape[1] != config.quantum_depth:
        raise ValueError(
            f"vision_iqml expects token count == quantum_depth, got {x.shape[1]} and {config.quantum_depth}"
        )
    hidden_sequence = _lstm_forward(params["lstm"], x, config.hidden_dim)
    theta = _quantum_theta(params, hidden_sequence, config)
    quantum_features = run_pqc_batch(theta, _quantum_config(config, depth=theta.shape[1]))
    normalized_features = _layer_norm(quantum_features)
    final_hidden = hidden_sequence[:, -1, :]
    fusion_mode = _normalize_fusion_mode(config.fusion_mode)
    if fusion_mode == "logit_residual":
        classical_logits = _linear(params["classical_head"], final_hidden)
        quantum_hidden = jax.nn.relu(_linear(params["quantum_head"]["hidden"], normalized_features))
        quantum_logits = _linear(params["quantum_head"]["out"], quantum_hidden)
        logits = classical_logits + quantum_logits
    elif fusion_mode == "feature_concat":
        joint_features = jnp.concatenate([_layer_norm(final_hidden), normalized_features], axis=-1)
        fusion_hidden = jax.nn.relu(_linear(params["fusion_head"]["hidden"], joint_features))
        logits = _linear(params["fusion_head"]["out"], fusion_hidden)
        classical_logits = jnp.zeros_like(logits)
        quantum_logits = jnp.zeros_like(logits)
    else:
        raise ValueError("fusion_mode must be 'logit_residual' or 'feature_concat'")
    return VisionIQMLOutput(
        logits=logits,
        classical_logits=classical_logits,
        quantum_logits=quantum_logits,
        quantum_features=quantum_features,
        normalized_quantum_features=normalized_features,
        theta=theta,
        hidden_sequence=hidden_sequence,
    )


def vision_iqml_logits_from_quantum_features(
    params: Params,
    hidden_sequence: jnp.ndarray,
    quantum_features: jnp.ndarray,
    config: VisionIQMLConfig,
) -> jnp.ndarray:
    normalized_features = _layer_norm(quantum_features)
    final_hidden = hidden_sequence[:, -1, :]
    fusion_mode = _normalize_fusion_mode(config.fusion_mode)
    if fusion_mode == "logit_residual":
        classical_logits = _linear(params["classical_head"], final_hidden)
        quantum_hidden = jax.nn.relu(_linear(params["quantum_head"]["hidden"], normalized_features))
        quantum_logits = _linear(params["quantum_head"]["out"], quantum_hidden)
        return classical_logits + quantum_logits
    if fusion_mode == "feature_concat":
        joint_features = jnp.concatenate([_layer_norm(final_hidden), normalized_features], axis=-1)
        fusion_hidden = jax.nn.relu(_linear(params["fusion_head"]["hidden"], joint_features))
        return _linear(params["fusion_head"]["out"], fusion_hidden)
    raise ValueError("fusion_mode must be 'logit_residual' or 'feature_concat'")


def vision_lstm_loss(params: Params, x: jnp.ndarray, y: jnp.ndarray, config: VisionLSTMConfig) -> jnp.ndarray:
    return _cross_entropy(vision_lstm_forward(params, x, config).logits, y)


def vision_iqml_loss(params: Params, x: jnp.ndarray, y: jnp.ndarray, config: VisionIQMLConfig) -> jnp.ndarray:
    return _cross_entropy(vision_iqml_forward(params, x, config).logits, y)


def _quantum_theta(params: Params, hidden_sequence: jnp.ndarray, config: VisionIQMLConfig) -> jnp.ndarray:
    theta_source = _normalize_theta_source(config.theta_source)
    if theta_source == "hidden_direct":
        return _normalize_hidden_direct_theta(hidden_sequence, config)
    if theta_source == "hidden":
        return jnp.pi * jnp.tanh(_linear(params["quantum_encoder"], hidden_sequence))
    raise ValueError("theta_source must be 'hidden' or 'hidden_direct'")


def _normalize_hidden_direct_theta(hidden_sequence: jnp.ndarray, config: VisionIQMLConfig) -> jnp.ndarray:
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


def _quantum_config(config: VisionIQMLConfig, depth: int | None = None) -> QuantumCircuitConfig:
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


def _normalize_theta_source(theta_source: str) -> str:
    key = theta_source.lower().replace("-", "_")
    if key in {"hidden_direct", "lstm_hidden", "hidden_state"}:
        return "hidden_direct"
    return key


def _normalize_fusion_mode(fusion_mode: str) -> str:
    key = fusion_mode.lower().replace("-", "_")
    if key in {"logit_residual", "residual_logits", "logits", "residual"}:
        return "logit_residual"
    if key in {"feature_concat", "concat", "feature_fusion", "joint_features"}:
        return "feature_concat"
    return key


def accuracy_from_logits(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.argmax(logits, axis=-1).astype(y.dtype) == y)


def _cross_entropy(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, y[:, None], axis=-1))


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
    axis: int | tuple[int, ...] = -1,
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
