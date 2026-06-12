from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from iqml.models.params import count_parameters
from iqml.vqe.ansatz import VQEConfig, build_tensorcircuit_hamiltonian
from iqml.vqe.chemistry import PauliHamiltonian
from iqml.vqe.features import build_lstm_vqe_inputs
from iqml.vqe.models import (
    Params,
    independent_vqe_energy,
    init_independent_vqe_params,
    init_lstm_vqe_params,
    lstm_vqe_energy,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VQETrainState:
    params: Params
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    step: int = 0

    def tree_flatten(self):
        children = (self.params, self.opt_state, jnp.asarray(self.step, dtype=jnp.int32))
        return children, self.optimizer

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, opt_state, step = children
        return cls(params=params, optimizer=aux_data, opt_state=opt_state, step=step)


@dataclass(frozen=True)
class VQERunResult:
    params: Params
    history: list[dict[str, float]]
    metrics: dict[str, float]


def run_vqe_training(
    method: str,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
    spacing: float,
    seed: int,
    epochs: int,
    learning_rate: float,
    lstm_hidden_dim: int = 16,
    lstm_input_mode: str = "physical",
    lstm_token_dim: int = 4,
    lstm_output_mode: str = "base_residual",
    lstm_base_scale: float = 0.05,
    exact_energy: float | None = None,
    grad_clip_norm: float | None = None,
) -> VQERunResult:
    optimizer = optax.adam(learning_rate)
    method_key = method.lower()
    hamiltonian_operator = build_tensorcircuit_hamiltonian(hamiltonian)

    if method_key == "independent":
        params = init_independent_vqe_params(jax.random.PRNGKey(seed), config)
        energy_fn: Callable[[Params], jnp.ndarray] = lambda p: independent_vqe_energy(
            p,
            hamiltonian,
            config,
            hamiltonian_operator=hamiltonian_operator,
        )
    elif method_key == "lstm":
        if lstm_input_mode == "physical":
            features = build_lstm_vqe_inputs(hamiltonian=hamiltonian, depth=config.depth, spacing=spacing)
            feature_dim = features.shape[-1]
        elif lstm_input_mode == "learned_token":
            features = None
            feature_dim = 0
        else:
            raise ValueError("lstm_input_mode must be 'physical' or 'learned_token'")
        params = init_lstm_vqe_params(
            jax.random.PRNGKey(seed),
            feature_dim=feature_dim,
            hidden_dim=lstm_hidden_dim,
            config=config,
            input_mode=lstm_input_mode,
            token_dim=lstm_token_dim,
            output_mode=lstm_output_mode,
            base_scale=lstm_base_scale,
        )
        energy_fn = lambda p: lstm_vqe_energy(
            p,
            features,
            hamiltonian,
            config,
            input_mode=lstm_input_mode,
            output_mode=lstm_output_mode,
            hamiltonian_operator=hamiltonian_operator,
        )
    else:
        raise ValueError("method must be 'independent' or 'lstm'")

    state = VQETrainState(params=params, optimizer=optimizer, opt_state=optimizer.init(params))
    train_step = _make_vqe_train_step(energy_fn, optimizer, grad_clip_norm)
    history = []
    for _ in range(epochs):
        state, metrics = train_step(state)
        row = {
            "energy": float(metrics["energy"]),
            "grad_norm": float(metrics["grad_norm"]),
        }
        if exact_energy is not None:
            row["energy_error"] = float(metrics["energy"] - exact_energy)
        history.append(row)

    final_energy = history[-1]["energy"] if history else float(energy_fn(state.params))
    result_metrics = {
        "energy": final_energy,
        "grad_norm": history[-1]["grad_norm"] if history else 0.0,
        "num_parameters": float(count_parameters(state.params)),
    }
    if exact_energy is not None:
        result_metrics["exact_energy"] = float(exact_energy)
        result_metrics["energy_error"] = float(final_energy - exact_energy)
        result_metrics["abs_energy_error"] = abs(float(final_energy - exact_energy))
    return VQERunResult(
        params=replace(state, step=int(state.step)).params,
        history=history,
        metrics=result_metrics,
    )


def _make_vqe_train_step(
    energy_fn: Callable[[Params], jnp.ndarray],
    optimizer: optax.GradientTransformation,
    grad_clip_norm: float | None,
):
    def _step(state: VQETrainState) -> tuple[VQETrainState, dict[str, jnp.ndarray]]:
        energy, grads = jax.value_and_grad(energy_fn)(state.params)
        raw_grad_norm = _tree_l2_norm(grads)
        clipped_grads = _clip_grads(grads, raw_grad_norm, grad_clip_norm)
        grad_norm = _tree_l2_norm(clipped_grads)
        updates, opt_state = optimizer.update(clipped_grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return (
            VQETrainState(
                params=params,
                optimizer=state.optimizer,
                opt_state=opt_state,
                step=state.step + jnp.asarray(1, dtype=jnp.int32),
            ),
            {
                "energy": energy,
                "grad_norm": grad_norm,
                "raw_grad_norm": raw_grad_norm,
            },
        )

    return jax.jit(_step)


def _tree_l2_norm(tree: Params) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.asarray([jnp.sum(jnp.asarray(leaf) ** 2) for leaf in leaves])))


def _clip_grads(
    grads: Params,
    grad_norm: jnp.ndarray,
    grad_clip_norm: float | None,
) -> Params:
    if grad_clip_norm is None or grad_clip_norm <= 0:
        return grads
    clip = jnp.asarray(grad_clip_norm, dtype=jnp.float32)
    scale = jnp.minimum(1.0, clip / (grad_norm + 1e-12))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)
