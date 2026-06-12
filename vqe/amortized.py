from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from iqml.vqe.ansatz import VQEConfig, he_vqe_energy
from iqml.vqe.chemistry import PauliHamiltonian

Params = dict[str, Any]


@dataclass(frozen=True)
class AmortizedVQEModelConfig:
    molecule_feature_dim: int
    qubit_feature_dim: int
    context_dim: int = 16
    layer_hidden_dim: int = 16
    head_hidden_dim: int = 32
    residual_scale: float = 0.1


def build_hchain_molecule_sequence(num_atoms: int, spacing: float) -> jnp.ndarray:
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    atom_index = jnp.arange(num_atoms, dtype=jnp.float32)
    denom = float(max(num_atoms - 1, 1))
    normalized_index = atom_index / denom
    centered_position = (atom_index - 0.5 * denom) * float(spacing)
    max_position = max(float(num_atoms - 1) * float(spacing), 1.0)
    left_bond = jnp.where(atom_index == 0, 0.0, float(spacing))
    right_bond = jnp.where(atom_index == num_atoms - 1, 0.0, float(spacing))
    edge_flag = jnp.where((atom_index == 0) | (atom_index == num_atoms - 1), 1.0, 0.0)
    hydrogen_charge = jnp.ones((num_atoms,), dtype=jnp.float32)
    return jnp.stack(
        [
            normalized_index,
            hydrogen_charge,
            left_bond,
            right_bond,
            centered_position / max_position,
            edge_flag,
        ],
        axis=-1,
    ).astype(jnp.float32)


def build_hchain_qubit_sequence(num_atoms: int, spacing: float) -> jnp.ndarray:
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    num_qubits = 2 * num_atoms
    qubit_index = jnp.arange(num_qubits, dtype=jnp.float32)
    atom_index = jnp.floor_divide(qubit_index.astype(jnp.int32), 2).astype(jnp.float32)
    spin = jnp.mod(qubit_index, 2.0)
    denom = float(max(num_atoms - 1, 1))
    normalized_atom = atom_index / denom
    centered_position = (atom_index - 0.5 * denom) * float(spacing)
    max_position = max(float(num_atoms - 1) * float(spacing), 1.0)
    return jnp.stack(
        [
            qubit_index / float(max(num_qubits - 1, 1)),
            normalized_atom,
            spin,
            jnp.full((num_qubits,), float(spacing), dtype=jnp.float32),
            centered_position / max_position,
        ],
        axis=-1,
    ).astype(jnp.float32)


def init_amortized_vqe_params(
    key: jax.Array,
    config: AmortizedVQEModelConfig,
) -> Params:
    keys = jax.random.split(key, 7)
    return {
        "molecule_encoder": {
            "W": _glorot(keys[0], (config.molecule_feature_dim, config.context_dim)),
            "b": jnp.zeros((config.context_dim,), dtype=jnp.float32),
        },
        "qubit_encoder": {
            "W": _glorot(keys[1], (config.qubit_feature_dim, config.context_dim)),
            "b": jnp.zeros((config.context_dim,), dtype=jnp.float32),
        },
        "layer_lstm": {
            "W": _glorot(keys[2], (2 * config.context_dim + config.layer_hidden_dim, 4 * config.layer_hidden_dim)),
            "b": jnp.zeros((4 * config.layer_hidden_dim,), dtype=jnp.float32),
        },
        "head_hidden": {
            "W": _glorot(keys[3], (config.layer_hidden_dim + 2 * config.context_dim + 1, config.head_hidden_dim)),
            "b": jnp.zeros((config.head_hidden_dim,), dtype=jnp.float32),
        },
        "head_out": {
            "W": _glorot(keys[4], (config.head_hidden_dim, 2)),
            "b": jnp.zeros((2,), dtype=jnp.float32),
        },
        "layer_token": {
            "W": 0.05 * jax.random.normal(keys[5], (config.context_dim,), dtype=jnp.float32),
        },
        "theta_base": {
            "W": 0.05 * jax.random.normal(keys[6], (2,), dtype=jnp.float32),
        },
    }


def amortized_vqe_theta(
    params: Params,
    molecule_features: jnp.ndarray,
    qubit_features: jnp.ndarray,
    vqe_config: VQEConfig,
    model_config: AmortizedVQEModelConfig,
) -> jnp.ndarray:
    if molecule_features.shape[-1] != model_config.molecule_feature_dim:
        raise ValueError("molecule feature dimension mismatch")
    if qubit_features.shape != (vqe_config.num_qubits, model_config.qubit_feature_dim):
        raise ValueError(
            "qubit_features must have shape "
            f"({vqe_config.num_qubits}, {model_config.qubit_feature_dim})"
        )

    molecule_context = _encode_sequence(params["molecule_encoder"], molecule_features)
    qubit_context = jnp.tanh(qubit_features @ params["qubit_encoder"]["W"] + params["qubit_encoder"]["b"])

    layer_inputs = jnp.tile(
        jnp.concatenate([molecule_context, params["layer_token"]["W"]])[None, :],
        (vqe_config.depth, 1),
    )
    layer_hidden = _lstm_forward(params["layer_lstm"], layer_inputs, model_config.layer_hidden_dim)
    layer_index = jnp.arange(vqe_config.depth, dtype=jnp.float32) / float(max(vqe_config.depth - 1, 1))

    def layer_to_theta(hidden_l: jnp.ndarray, index_l: jnp.ndarray) -> jnp.ndarray:
        per_qubit_hidden = jnp.tile(hidden_l[None, :], (vqe_config.num_qubits, 1))
        per_molecule = jnp.tile(molecule_context[None, :], (vqe_config.num_qubits, 1))
        per_index = jnp.full((vqe_config.num_qubits, 1), index_l, dtype=jnp.float32)
        head_input = jnp.concatenate([per_qubit_hidden, per_molecule, qubit_context, per_index], axis=-1)
        hidden = jax.nn.relu(head_input @ params["head_hidden"]["W"] + params["head_hidden"]["b"])
        return hidden @ params["head_out"]["W"] + params["head_out"]["b"]

    raw = jax.vmap(layer_to_theta)(layer_hidden, layer_index)
    deltas = model_config.residual_scale * jnp.tanh(raw)
    theta = jnp.cumsum(deltas, axis=0) + params["theta_base"]["W"]
    return jnp.clip(theta, -jnp.pi, jnp.pi)


def amortized_vqe_energy(
    params: Params,
    molecule_features: jnp.ndarray,
    qubit_features: jnp.ndarray,
    hamiltonian: PauliHamiltonian,
    vqe_config: VQEConfig,
    model_config: AmortizedVQEModelConfig,
    hamiltonian_operator: Any | None = None,
) -> jnp.ndarray:
    theta = amortized_vqe_theta(params, molecule_features, qubit_features, vqe_config, model_config)
    return he_vqe_energy(theta, hamiltonian, vqe_config, hamiltonian_operator)


def _encode_sequence(params: Params, features: jnp.ndarray) -> jnp.ndarray:
    encoded = jnp.tanh(features @ params["W"] + params["b"])
    return jnp.mean(encoded, axis=0)


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
