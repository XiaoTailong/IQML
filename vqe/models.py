from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from iqml.vqe.ansatz import (
    VQEConfig,
    ansatz_layer_parameter_size,
    ansatz_parameter_shape,
    ansatz_parameter_size,
    is_xyz_hamiltonian_ansatz,
    vqe_energy,
)
from iqml.vqe.chemistry import PauliHamiltonian
from iqml.vqe.compact_xyz import CompactXYZHamiltonian, compact_xyz_energy

Params = dict[str, Any]


def init_independent_vqe_params(
    key: jax.Array,
    config: VQEConfig,
    scale: float = 0.05,
) -> Params:
    return {
        "theta": scale
        * jax.random.normal(
            key,
            ansatz_parameter_shape(config),
            dtype=jnp.float32,
        )
    }


def independent_vqe_energy(
    params: Params,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
    hamiltonian_operator: Any | None = None,
) -> jnp.ndarray:
    if isinstance(hamiltonian_operator, CompactXYZHamiltonian) and is_xyz_hamiltonian_ansatz(config):
        return compact_xyz_energy(
            params["theta"],
            hamiltonian_operator,
            hamiltonian_gate_scale=config.hamiltonian_gate_scale,
        )
    return vqe_energy(params["theta"], hamiltonian, config, hamiltonian_operator)


def init_lstm_vqe_params(
    key: jax.Array,
    feature_dim: int,
    hidden_dim: int,
    config: VQEConfig,
    theta_mode: str = "residual",
    residual_scale: float = 0.1,
    input_mode: str = "physical",
    token_dim: int = 4,
    output_mode: str = "base_residual",
    base_scale: float = 0.05,
    hidden_theta_scale_trainable: bool = False,
    hidden_theta_scale_init: float | None = None,
) -> Params:
    del theta_mode, residual_scale
    keys = jax.random.split(key, 4)
    input_dim = _resolve_lstm_input_dim(feature_dim, input_mode, token_dim)
    layer_parameter_size = ansatz_layer_parameter_size(config)
    hidden_direct = _normalize_theta_mode(config.theta_mode) == "hidden_direct"
    if hidden_direct and hidden_dim != layer_parameter_size:
        raise ValueError(
            "theta_mode='hidden_direct' requires hidden_dim to equal the VQE "
            f"layer parameter count ({layer_parameter_size}), got {hidden_dim}"
        )
    _validate_lstm_output_mode(output_mode)
    params = {
        "lstm": {
            "W": _glorot(keys[0], (input_dim + hidden_dim, 4 * hidden_dim)),
            "b": jnp.zeros((4 * hidden_dim,), dtype=jnp.float32),
        },
    }
    if not hidden_direct:
        params["head"] = {
            "W": _glorot(keys[1], (hidden_dim, layer_parameter_size)),
            "b": jnp.zeros((layer_parameter_size,), dtype=jnp.float32),
        }
    if input_mode == "learned_token":
        params["input"] = {
            "token": 0.05 * jax.random.normal(keys[2], (token_dim,), dtype=jnp.float32),
        }
    if output_mode == "base_residual":
        params["theta_base"] = base_scale * jax.random.normal(
            keys[3],
            ansatz_parameter_shape(config),
            dtype=jnp.float32,
        )
    if hidden_direct and hidden_theta_scale_trainable:
        init_scale = config.hidden_theta_scale if hidden_theta_scale_init is None else hidden_theta_scale_init
        params["theta_scale"] = {
            "value": jnp.asarray(init_scale, dtype=jnp.float32),
        }
    return params


def init_mlp_vqe_params(
    key: jax.Array,
    token_dim: int,
    head_hidden_dim: int,
    config: VQEConfig,
    output_mode: str = "base_residual",
    base_scale: float = 0.05,
    feature_dim: int = 0,
    input_mode: str = "learned_token",
) -> Params:
    if input_mode == "learned_token" and token_dim <= 0:
        raise ValueError("token_dim must be positive")
    if input_mode == "physical" and feature_dim <= 0:
        raise ValueError("feature_dim must be positive for physical MLP-VQE inputs")
    if input_mode not in {"learned_token", "physical"}:
        raise ValueError("input_mode must be 'learned_token' or 'physical'")
    if head_hidden_dim <= 0:
        raise ValueError("head_hidden_dim must be positive")
    _validate_lstm_output_mode(output_mode)
    keys = jax.random.split(key, 4)
    theta_size = ansatz_parameter_size(config)
    input_dim = token_dim if input_mode == "learned_token" else feature_dim
    output_dim = theta_size if input_mode == "learned_token" else ansatz_layer_parameter_size(config)
    params = {
        "mlp": {
            "W1": _glorot(keys[1], (input_dim, head_hidden_dim)),
            "b1": jnp.zeros((head_hidden_dim,), dtype=jnp.float32),
            "W2": _glorot(keys[2], (head_hidden_dim, output_dim)),
            "b2": jnp.zeros((output_dim,), dtype=jnp.float32),
        },
    }
    if input_mode == "learned_token":
        params["input"] = {
            "token": 0.05 * jax.random.normal(keys[0], (token_dim,), dtype=jnp.float32),
        }
    if output_mode == "base_residual":
        params["theta_base"] = base_scale * jax.random.normal(
            keys[3],
            ansatz_parameter_shape(config),
            dtype=jnp.float32,
        )
    return params


def mlp_vqe_energy(
    params: Params,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
    output_mode: str = "base_residual",
    hamiltonian_operator: Any | None = None,
    features: jnp.ndarray | None = None,
    input_mode: str = "learned_token",
) -> jnp.ndarray:
    theta = mlp_vqe_theta(params, config, output_mode=output_mode, features=features, input_mode=input_mode)
    if isinstance(hamiltonian_operator, CompactXYZHamiltonian) and is_xyz_hamiltonian_ansatz(config):
        return compact_xyz_energy(
            theta,
            hamiltonian_operator,
            hamiltonian_gate_scale=config.hamiltonian_gate_scale,
        )
    return vqe_energy(theta, hamiltonian, config, hamiltonian_operator)


def mlp_vqe_theta(
    params: Params,
    config: VQEConfig,
    output_mode: str = "base_residual",
    features: jnp.ndarray | None = None,
    input_mode: str = "learned_token",
) -> jnp.ndarray:
    _validate_lstm_output_mode(output_mode)
    if input_mode == "learned_token":
        if "input" not in params or "token" not in params["input"]:
            raise ValueError("MLP-VQE params must contain input.token")
        hidden = jnp.tanh(params["input"]["token"] @ params["mlp"]["W1"] + params["mlp"]["b1"])
        raw = hidden @ params["mlp"]["W2"] + params["mlp"]["b2"]
        layer_values = raw.reshape(ansatz_parameter_shape(config))
    elif input_mode == "physical":
        if features is None:
            raise ValueError("features are required for physical MLP-VQE inputs")
        if features.shape[0] != config.depth:
            raise ValueError(f"features depth {features.shape[0]} must equal config.depth {config.depth}")
        hidden = jnp.tanh(features @ params["mlp"]["W1"] + params["mlp"]["b1"])
        raw = hidden @ params["mlp"]["W2"] + params["mlp"]["b2"]
        layer_values = raw.reshape(ansatz_parameter_shape(config))
    else:
        raise ValueError("input_mode must be 'learned_token' or 'physical'")
    if config.theta_mode == "direct":
        return jnp.pi * jnp.tanh(layer_values)
    if config.theta_mode != "residual":
        raise ValueError("theta_mode must be 'residual' or 'direct'")
    theta = config.residual_scale * jnp.tanh(layer_values)
    if input_mode == "physical":
        theta = jnp.cumsum(theta, axis=0)
    if output_mode == "base_residual":
        if "theta_base" not in params:
            raise ValueError("base_residual MLP-VQE params must contain theta_base")
        theta = params["theta_base"] + theta
    return jnp.clip(theta, -jnp.pi, jnp.pi)


def lstm_vqe_energy(
    params: Params,
    features: jnp.ndarray | None,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
    input_mode: str = "physical",
    output_mode: str = "base_residual",
    hamiltonian_operator: Any | None = None,
) -> jnp.ndarray:
    theta = lstm_vqe_theta(params, features, config, input_mode=input_mode, output_mode=output_mode)
    if isinstance(hamiltonian_operator, CompactXYZHamiltonian) and is_xyz_hamiltonian_ansatz(config):
        return compact_xyz_energy(
            theta,
            hamiltonian_operator,
            hamiltonian_gate_scale=config.hamiltonian_gate_scale,
        )
    return vqe_energy(theta, hamiltonian, config, hamiltonian_operator)


def lstm_vqe_theta(
    params: Params,
    features: jnp.ndarray | None,
    config: VQEConfig,
    input_mode: str = "physical",
    output_mode: str = "base_residual",
) -> jnp.ndarray:
    _validate_lstm_output_mode(output_mode)
    features = _build_lstm_inputs(params, features, config, input_mode)
    hidden_dim = params["lstm"]["b"].shape[0] // 4
    hidden = _lstm_forward(params["lstm"], features, hidden_dim)
    theta_mode = _normalize_theta_mode(config.theta_mode)
    if theta_mode == "hidden_direct":
        expected_shape = ansatz_parameter_shape(config)
        expected_hidden_shape = (config.depth, ansatz_layer_parameter_size(config))
        if hidden.shape != expected_hidden_shape:
            raise ValueError(
                "theta_mode='hidden_direct' requires the LSTM hidden sequence "
                f"to have shape {expected_hidden_shape}, got {hidden.shape}"
            )
        layer_values = hidden.reshape(expected_shape)
        normalized = _normalize_hidden_direct_theta(layer_values, config)
        if "theta_scale" in params:
            scale = jnp.asarray(params["theta_scale"]["value"], dtype=hidden.dtype)
        else:
            scale = jnp.asarray(config.hidden_theta_scale, dtype=hidden.dtype)
        return scale * normalized
    raw = hidden @ params["head"]["W"] + params["head"]["b"]
    layer_values = raw.reshape(ansatz_parameter_shape(config))
    if theta_mode == "direct":
        return jnp.pi * jnp.tanh(layer_values)
    if theta_mode != "residual":
        raise ValueError("theta_mode must be 'residual', 'direct', or 'hidden_direct'")
    deltas = config.residual_scale * jnp.tanh(layer_values)
    theta = jnp.cumsum(deltas, axis=0)
    if output_mode == "base_residual":
        if "theta_base" not in params:
            raise ValueError("base_residual LSTM-VQE params must contain theta_base")
        theta = params["theta_base"] + theta
    return jnp.clip(theta, -jnp.pi, jnp.pi)


def _validate_lstm_output_mode(output_mode: str) -> None:
    if output_mode not in {"generator", "base_residual"}:
        raise ValueError("output_mode must be 'generator' or 'base_residual'")


def _resolve_lstm_input_dim(feature_dim: int, input_mode: str, token_dim: int) -> int:
    if input_mode == "physical":
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for physical LSTM-VQE inputs")
        return feature_dim
    if input_mode == "learned_token":
        if token_dim <= 0:
            raise ValueError("token_dim must be positive for learned_token LSTM-VQE inputs")
        return token_dim
    raise ValueError("input_mode must be 'physical' or 'learned_token'")


def _build_lstm_inputs(
    params: Params,
    features: jnp.ndarray | None,
    config: VQEConfig,
    input_mode: str,
) -> jnp.ndarray:
    if input_mode == "physical":
        if features is None:
            raise ValueError("features are required for physical LSTM-VQE inputs")
        if features.shape[0] != config.depth:
            raise ValueError(f"features depth {features.shape[0]} must equal config.depth {config.depth}")
        return features
    if input_mode == "learned_token":
        if "input" not in params or "token" not in params["input"]:
            raise ValueError("learned_token LSTM-VQE params must contain input.token")
        token = params["input"]["token"]
        return jnp.tile(token[None, :], (config.depth, 1))
    raise ValueError("input_mode must be 'physical' or 'learned_token'")


def _normalize_theta_mode(theta_mode: str) -> str:
    key = theta_mode.lower().replace("-", "_")
    if key in {"hidden_direct", "lstm_hidden", "hidden_state", "direct_hidden"}:
        return "hidden_direct"
    return key


def _centered_rms_norm(
    x: jnp.ndarray,
    *,
    eps: float,
    axis: tuple[int, ...],
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=axis, keepdims=True)
    centered = x - mean
    rms = jnp.sqrt(jnp.mean(centered**2, axis=axis, keepdims=True) + eps)
    return centered / rms


def _normalize_hidden_direct_theta(x: jnp.ndarray, config: VQEConfig) -> jnp.ndarray:
    axis = tuple(range(1, x.ndim))
    mode = config.hidden_theta_norm.lower().replace("-", "_")
    if mode in {"rms", "centered_rms", "standard"}:
        return _centered_rms_norm(x, eps=config.hidden_theta_eps, axis=axis)
    if mode in {"minmax_symmetric", "symmetric_minmax", "minmax"}:
        x_min = jnp.min(x, axis=axis, keepdims=True)
        x_max = jnp.max(x, axis=axis, keepdims=True)
        span = x_max - x_min
        unit = jnp.where(
            span > config.hidden_theta_eps,
            (x - x_min) / (span + config.hidden_theta_eps),
            0.5,
        )
        return 2.0 * unit - 1.0
    if mode in {"minmax_positive", "positive_minmax"}:
        x_min = jnp.min(x, axis=axis, keepdims=True)
        x_max = jnp.max(x, axis=axis, keepdims=True)
        span = x_max - x_min
        return jnp.where(
            span > config.hidden_theta_eps,
            (x - x_min) / (span + config.hidden_theta_eps),
            0.5,
        )
    raise ValueError(
        "hidden_theta_norm must be 'rms', 'minmax_symmetric', or 'minmax_positive'"
    )


def _lstm_forward(params: Params, x: jnp.ndarray, hidden_dim: int) -> jnp.ndarray:
    dtype = x.dtype
    h0 = jnp.zeros((hidden_dim,), dtype=dtype)
    c0 = jnp.zeros((hidden_dim,), dtype=dtype)

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
