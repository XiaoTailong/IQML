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
class TransformerIQMLConfig:
    input_dim: int
    output_dim: int
    seq_len: int = 48
    patch_len: int = 8
    patch_stride: int = 4
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    num_qubits: int = 8
    circuit_type: str = "he"
    entanglement: str = "linear"
    observables: str = "z,zz"
    readout_stride: int = 0
    fusion_hidden_dim: int = 24
    fusion_mode: str = "residual_add"
    residual_init_scale: float = 0.01
    hidden_theta_scale: float = math.pi
    hidden_theta_eps: float = 1e-6
    hidden_theta_norm: str = "layer_rms"


@dataclass(frozen=True)
class TransformerIQMLOutput:
    prediction: jnp.ndarray
    base_prediction: jnp.ndarray
    quantum_residual: jnp.ndarray
    transformer_tokens: jnp.ndarray
    quantum_features: jnp.ndarray
    normalized_quantum_features: jnp.ndarray
    theta: jnp.ndarray
    joint_features: jnp.ndarray


def init_transformer_iqml_params(key: jax.Array, config: TransformerIQMLConfig) -> Params:
    _validate_config(config)
    depth = patch_count(config)
    quantum_config = _quantum_config(config, depth=depth)
    quantum_parameter_count = parameter_count(quantum_config)
    quantum_feature_count = feature_count(quantum_config, depth=depth)
    num_keys = 12 + config.num_layers * 10
    keys = list(jax.random.split(key, num_keys))

    params: Params = {
        "patch_embed": {
            "W": _glorot(keys.pop(0), (config.input_dim * config.patch_len, config.d_model)),
            "b": jnp.zeros((config.d_model,), dtype=jnp.float32),
        },
        "position_embedding": 0.02 * jax.random.normal(keys.pop(0), (depth, config.d_model), dtype=jnp.float32),
        "layers": [],
        "base_head_norm": {
            "gamma": jnp.ones((depth * config.d_model,), dtype=jnp.float32),
            "beta": jnp.zeros((depth * config.d_model,), dtype=jnp.float32),
        },
        "quantum_encoder": {
            "W": _glorot(keys.pop(0), (config.d_model, quantum_parameter_count)),
            "b": jnp.zeros((quantum_parameter_count,), dtype=jnp.float32),
        },
    }
    fusion_mode = _normalize_fusion_mode(config.fusion_mode)
    if fusion_mode == "residual_add":
        params["base_head"] = {
            "hidden": {
                "W": _glorot(keys.pop(0), (depth * config.d_model, config.ff_dim)),
                "b": jnp.zeros((config.ff_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": _glorot(keys.pop(0), (config.ff_dim, config.output_dim)),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        }
        params["quantum_head"] = {
            "hidden": {
                "W": _glorot(keys.pop(0), (quantum_feature_count, config.fusion_hidden_dim)),
                "b": jnp.zeros((config.fusion_hidden_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": config.residual_init_scale
                * jax.random.normal(keys.pop(0), (config.fusion_hidden_dim, config.output_dim), dtype=jnp.float32),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        }
    elif fusion_mode == "feature_concat":
        joint_dim = depth * config.d_model + quantum_feature_count
        params["joint_norm"] = {
            "gamma": jnp.ones((joint_dim,), dtype=jnp.float32),
            "beta": jnp.zeros((joint_dim,), dtype=jnp.float32),
        }
        params["joint_head"] = {
            "hidden": {
                "W": _glorot(keys.pop(0), (joint_dim, config.ff_dim)),
                "b": jnp.zeros((config.ff_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": _glorot(keys.pop(0), (config.ff_dim, config.output_dim)),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        }
    elif fusion_mode == "feature_film":
        params["base_head"] = {
            "hidden": {
                "W": _glorot(keys.pop(0), (depth * config.d_model, config.ff_dim)),
                "b": jnp.zeros((config.ff_dim,), dtype=jnp.float32),
            },
        }
        params["film_head"] = {
            "scale": {
                "W": 0.05 * jax.random.normal(keys.pop(0), (quantum_feature_count, config.ff_dim), dtype=jnp.float32),
                "b": jnp.zeros((config.ff_dim,), dtype=jnp.float32),
            },
            "shift": {
                "W": _glorot(keys.pop(0), (quantum_feature_count, config.ff_dim)),
                "b": jnp.zeros((config.ff_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": _glorot(keys.pop(0), (config.ff_dim, config.output_dim)),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        }
    for _ in range(config.num_layers):
        params["layers"].append(
            {
                "attn_norm": _init_norm(config.d_model),
                "ff_norm": _init_norm(config.d_model),
                "qkv": {
                    "W": _glorot(keys.pop(0), (config.d_model, 3 * config.d_model)),
                    "b": jnp.zeros((3 * config.d_model,), dtype=jnp.float32),
                },
                "attn_out": {
                    "W": _glorot(keys.pop(0), (config.d_model, config.d_model)),
                    "b": jnp.zeros((config.d_model,), dtype=jnp.float32),
                },
                "ff_hidden": {
                    "W": _glorot(keys.pop(0), (config.d_model, config.ff_dim)),
                    "b": jnp.zeros((config.ff_dim,), dtype=jnp.float32),
                },
                "ff_out": {
                    "W": _glorot(keys.pop(0), (config.ff_dim, config.d_model)),
                    "b": jnp.zeros((config.d_model,), dtype=jnp.float32),
                },
            }
        )
    return params


def transformer_iqml_forward(
    params: Params,
    x: jnp.ndarray,
    config: TransformerIQMLConfig,
) -> TransformerIQMLOutput:
    tokens = transformer_tokens(params, x, config)
    flat = tokens.reshape((tokens.shape[0], -1))
    flat = _param_layer_norm(params["base_head_norm"], flat)

    theta_raw = _linear(params["quantum_encoder"], tokens)
    theta = _normalize_theta(theta_raw, config)
    quantum_features = run_pqc_batch(theta, _quantum_config(config, depth=theta.shape[1]))
    normalized_features = _layer_norm(quantum_features)
    joint_features = jnp.concatenate([flat, normalized_features], axis=-1)
    fusion_mode = _normalize_fusion_mode(config.fusion_mode)
    if fusion_mode == "feature_concat":
        joint = _param_layer_norm(params["joint_norm"], joint_features)
        joint_hidden = jax.nn.gelu(_linear(params["joint_head"]["hidden"], joint))
        prediction = _linear(params["joint_head"]["out"], joint_hidden)
        base_prediction = jnp.zeros_like(prediction)
        quantum_residual = jnp.zeros_like(prediction)
    elif fusion_mode == "feature_film":
        base_hidden = jax.nn.gelu(_linear(params["base_head"]["hidden"], flat))
        scale = jnp.tanh(_linear(params["film_head"]["scale"], normalized_features))
        shift = _linear(params["film_head"]["shift"], normalized_features)
        fused_hidden = jax.nn.gelu(base_hidden * (1.0 + 0.1 * scale) + shift)
        base_prediction = _linear(params["film_head"]["out"], base_hidden)
        prediction = _linear(params["film_head"]["out"], fused_hidden)
        quantum_residual = prediction - base_prediction
    elif fusion_mode == "residual_add":
        base_hidden = jax.nn.gelu(_linear(params["base_head"]["hidden"], flat))
        base_prediction = _linear(params["base_head"]["out"], base_hidden)
        residual_hidden = jax.nn.relu(_linear(params["quantum_head"]["hidden"], normalized_features))
        quantum_residual = _linear(params["quantum_head"]["out"], residual_hidden)
        prediction = base_prediction + quantum_residual
    else:
        raise ValueError("fusion_mode must be 'residual_add', 'feature_concat', or 'feature_film'")
    return TransformerIQMLOutput(
        prediction=prediction,
        base_prediction=base_prediction,
        quantum_residual=quantum_residual,
        transformer_tokens=tokens,
        quantum_features=quantum_features,
        normalized_quantum_features=normalized_features,
        theta=theta,
        joint_features=joint_features,
    )


def transformer_tokens(params: Params, x: jnp.ndarray, config: TransformerIQMLConfig) -> jnp.ndarray:
    patches = _extract_patches(x, config)
    tokens = _linear(params["patch_embed"], patches) + params["position_embedding"][None, :, :]
    for layer in params["layers"]:
        attn_in = _param_layer_norm(layer["attn_norm"], tokens)
        tokens = tokens + _multi_head_attention(layer, attn_in, config)
        ff_in = _param_layer_norm(layer["ff_norm"], tokens)
        ff_hidden = jax.nn.gelu(_linear(layer["ff_hidden"], ff_in))
        tokens = tokens + _linear(layer["ff_out"], ff_hidden)
    return tokens


def transformer_iqml_mse_loss(
    params: Params,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: TransformerIQMLConfig,
) -> jnp.ndarray:
    output = transformer_iqml_forward(params, x, config)
    return jnp.mean((output.prediction - y) ** 2)


def patch_count(config: TransformerIQMLConfig) -> int:
    return 1 + (int(config.seq_len) - int(config.patch_len)) // int(config.patch_stride)


def _extract_patches(x: jnp.ndarray, config: TransformerIQMLConfig) -> jnp.ndarray:
    starts = range(0, config.seq_len - config.patch_len + 1, config.patch_stride)
    patches = [x[:, start : start + config.patch_len, :].reshape((x.shape[0], -1)) for start in starts]
    return jnp.stack(patches, axis=1)


def _multi_head_attention(layer: Params, x: jnp.ndarray, config: TransformerIQMLConfig) -> jnp.ndarray:
    batch, tokens, _ = x.shape
    head_dim = config.d_model // config.n_heads
    qkv = _linear(layer["qkv"], x)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape((batch, tokens, config.n_heads, head_dim))
    k = k.reshape((batch, tokens, config.n_heads, head_dim))
    v = v.reshape((batch, tokens, config.n_heads, head_dim))
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("bhts,bshd->bthd", weights, v)
    context = context.reshape((batch, tokens, config.d_model))
    return _linear(layer["attn_out"], context)


def _normalize_theta(theta_raw: jnp.ndarray, config: TransformerIQMLConfig) -> jnp.ndarray:
    mode = config.hidden_theta_norm.lower().replace("-", "_")
    scale = jnp.asarray(config.hidden_theta_scale, dtype=theta_raw.dtype)
    eps = config.hidden_theta_eps
    if mode in {"global_rms", "sample_rms", "window_rms", "rms"}:
        normalized = _centered_rms_norm(theta_raw, eps=eps, axis=(1, 2))
    elif mode in {"layer_rms", "token_rms", "per_layer_rms", "per_token_rms"}:
        normalized = _centered_rms_norm(theta_raw, eps=eps, axis=-1)
    elif mode in {"layer_minmax_symmetric", "token_minmax_symmetric", "per_layer_minmax_symmetric"}:
        normalized = _minmax_symmetric_norm(theta_raw, eps=eps, axis=-1)
    elif mode in {"layer_minmax_positive", "token_minmax_positive", "per_layer_minmax_positive"}:
        normalized = _minmax_positive_norm(theta_raw, eps=eps, axis=-1)
    else:
        raise ValueError(
            "hidden_theta_norm must be one of: global_rms, layer_rms, "
            "layer_minmax_symmetric, or layer_minmax_positive"
        )
    return scale * normalized


def _quantum_config(config: TransformerIQMLConfig, depth: int | None = None) -> QuantumCircuitConfig:
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


def _validate_config(config: TransformerIQMLConfig) -> None:
    if config.seq_len <= 0 or config.patch_len <= 0 or config.patch_stride <= 0:
        raise ValueError("seq_len, patch_len, and patch_stride must be positive")
    if config.patch_len > config.seq_len:
        raise ValueError("patch_len must be <= seq_len")
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if patch_count(config) <= 0:
        raise ValueError("patch configuration produced no tokens")
    _normalize_fusion_mode(config.fusion_mode)


def _normalize_fusion_mode(fusion_mode: str) -> str:
    key = fusion_mode.lower().replace("-", "_")
    if key in {"residual_add", "residual", "add", "logit_residual"}:
        return "residual_add"
    if key in {"feature_concat", "concat", "joint", "joint_head", "feature_fusion"}:
        return "feature_concat"
    if key in {"feature_film", "film", "gated", "gated_feature", "feature_gate"}:
        return "feature_film"
    raise ValueError("fusion_mode must be 'residual_add', 'feature_concat', or 'feature_film'")


def _linear(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["W"] + params["b"]


def _init_norm(width: int) -> Params:
    return {
        "gamma": jnp.ones((width,), dtype=jnp.float32),
        "beta": jnp.zeros((width,), dtype=jnp.float32),
    }


def _param_layer_norm(params: Params, x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    return params["gamma"] * _layer_norm(x, eps=eps) + params["beta"]


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
