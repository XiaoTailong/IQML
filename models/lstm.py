from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

Params = dict[str, Any]


@dataclass(frozen=True)
class LSTMConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    head_hidden_dim: int = 16


@dataclass(frozen=True)
class LSTMOutput:
    prediction: jnp.ndarray
    hidden_sequence: jnp.ndarray


def init_lstm_params(key: jax.Array, config: LSTMConfig) -> Params:
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
                "W": _glorot(keys[2], (config.head_hidden_dim, config.output_dim)),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        },
    }


def lstm_forward(params: Params, x: jnp.ndarray, config: LSTMConfig) -> LSTMOutput:
    hidden_sequence = _lstm_forward(params["lstm"], x, config.hidden_dim)
    last_hidden = hidden_sequence[:, -1, :]
    head_hidden = jax.nn.relu(_linear(params["head"]["hidden"], last_hidden))
    prediction = _linear(params["head"]["out"], head_hidden)
    return LSTMOutput(prediction=prediction, hidden_sequence=hidden_sequence)


def mse_loss(params: Params, x: jnp.ndarray, y: jnp.ndarray, config: LSTMConfig) -> jnp.ndarray:
    output = lstm_forward(params, x, config)
    return jnp.mean((output.prediction - y) ** 2)


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


def _glorot(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)
