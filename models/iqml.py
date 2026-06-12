from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from iqml.quantum.circuits import QuantumCircuitConfig, parameter_count, run_pqc_batch

Params = dict[str, Any]


@dataclass(frozen=True)
class IQMLConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_qubits: int
    circuit_type: str = "iqp"
    fusion_hidden_dim: int = 16
    entanglement: str = "linear"


@dataclass(frozen=True)
class IQMLOutput:
    prediction: jnp.ndarray
    classical_prediction: jnp.ndarray
    quantum_features: jnp.ndarray
    theta: jnp.ndarray
    hidden_sequence: jnp.ndarray


@dataclass(frozen=True)
class IQMLEncoding:
    classical_prediction: jnp.ndarray
    theta: jnp.ndarray
    hidden_sequence: jnp.ndarray


def init_iqml_params(key: jax.Array, config: IQMLConfig) -> Params:
    keys = jax.random.split(key, 8)
    quantum_config = _quantum_config(config)
    quantum_parameter_count = parameter_count(quantum_config)
    return {
        "lstm": {
            "W": _glorot(keys[0], (config.input_dim + config.hidden_dim, 4 * config.hidden_dim)),
            "b": jnp.zeros((4 * config.hidden_dim,), dtype=jnp.float32),
        },
        "classical_head": {
            "W": _glorot(keys[1], (config.hidden_dim, config.output_dim)),
            "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
        },
        "quantum_encoder": {
            "W": _glorot(keys[2], (config.hidden_dim, quantum_parameter_count)),
            "b": jnp.zeros((quantum_parameter_count,), dtype=jnp.float32),
        },
        "fusion": {
            "hidden": {
                "W": _glorot(keys[3], (config.output_dim + config.num_qubits, config.fusion_hidden_dim)),
                "b": jnp.zeros((config.fusion_hidden_dim,), dtype=jnp.float32),
            },
            "out": {
                "W": _glorot(keys[4], (config.fusion_hidden_dim, config.output_dim)),
                "b": jnp.zeros((config.output_dim,), dtype=jnp.float32),
            },
        },
    }


def iqml_encode(params: Params, x: jnp.ndarray, config: IQMLConfig) -> IQMLEncoding:
    hidden_sequence = _lstm_forward(params["lstm"], x, config.hidden_dim)
    last_hidden = hidden_sequence[:, -1, :]
    classical_prediction = _linear(params["classical_head"], last_hidden)
    theta = jnp.pi * jnp.tanh(_linear(params["quantum_encoder"], hidden_sequence))
    return IQMLEncoding(
        classical_prediction=classical_prediction,
        theta=theta,
        hidden_sequence=hidden_sequence,
    )


def iqml_predict_from_quantum_features(
    params: Params,
    classical_prediction: jnp.ndarray,
    quantum_features: jnp.ndarray,
) -> jnp.ndarray:
    fusion_input = jnp.concatenate([classical_prediction, quantum_features], axis=-1)
    fusion_hidden = jax.nn.relu(_linear(params["fusion"]["hidden"], fusion_input))
    return _linear(params["fusion"]["out"], fusion_hidden)


def iqml_forward(params: Params, x: jnp.ndarray, config: IQMLConfig) -> IQMLOutput:
    encoded = iqml_encode(params, x, config)
    quantum_features = run_pqc_batch(encoded.theta, _quantum_config(config))
    prediction = iqml_predict_from_quantum_features(
        params,
        encoded.classical_prediction,
        quantum_features,
    )
    return IQMLOutput(
        prediction=prediction,
        classical_prediction=encoded.classical_prediction,
        quantum_features=quantum_features,
        theta=encoded.theta,
        hidden_sequence=encoded.hidden_sequence,
    )


def mse_loss(params: Params, x: jnp.ndarray, y: jnp.ndarray, config: IQMLConfig) -> jnp.ndarray:
    output = iqml_forward(params, x, config)
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


def _quantum_config(config: IQMLConfig) -> QuantumCircuitConfig:
    return QuantumCircuitConfig(
        num_qubits=config.num_qubits,
        circuit_type=config.circuit_type,
        entanglement=config.entanglement,
    )


def _glorot(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)
