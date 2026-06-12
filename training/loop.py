from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

Params = dict[str, Any]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TrainState:
    params: Params
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    step: int = 0

    def tree_flatten(self):
        children = (self.params, self.opt_state, jnp.asarray(self.step, dtype=jnp.int32))
        aux_data = self.optimizer
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, opt_state, step = children
        return cls(params=params, optimizer=aux_data, opt_state=opt_state, step=step)


def create_train_state(
    params: Params,
    learning_rate: float = 1e-3,
    optimizer: optax.GradientTransformation | None = None,
) -> TrainState:
    optimizer = optimizer or optax.adam(learning_rate)
    return TrainState(params=params, optimizer=optimizer, opt_state=optimizer.init(params))


def train_step(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: Any,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    from iqml.models.iqml import iqml_forward, mse_loss

    loss_value, grads = jax.value_and_grad(mse_loss)(state.params, x, y, config)
    updates, opt_state = state.optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    output = iqml_forward(params, x, config)
    metrics = {
        "loss": loss_value,
        "mse": jnp.mean((output.prediction - y) ** 2),
    }
    return TrainState(params=params, optimizer=state.optimizer, opt_state=opt_state, step=state.step + 1), metrics


def make_train_step(
    config: Any,
    optimizer: optax.GradientTransformation,
):
    """Create a JIT-compiled train step for one static experiment config."""
    from iqml.models.iqml import iqml_forward, mse_loss

    def _step(
        state: TrainState,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        loss_value, grads = jax.value_and_grad(mse_loss)(state.params, x, y, config)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        output = iqml_forward(params, x, config)
        metrics = {
            "loss": loss_value,
            "mse": jnp.mean((output.prediction - y) ** 2),
        }
        return (
            TrainState(
                params=params,
                optimizer=state.optimizer,
                opt_state=opt_state,
                step=state.step + jnp.asarray(1, dtype=jnp.int32),
            ),
            metrics,
        )

    return jax.jit(_step)


def make_train_step_from_fns(
    loss_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    predict_fn: Callable[[Params, jnp.ndarray], jnp.ndarray],
    optimizer: optax.GradientTransformation,
    grad_clip_norm: float | None = None,
    metrics_fn: Callable[[jnp.ndarray, jnp.ndarray], dict[str, jnp.ndarray]] | None = None,
    compute_prediction_metrics: bool = True,
):
    """Create a JIT-compiled train step from model-specific callables."""

    def _step(
        state: TrainState,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        loss_value, grads = jax.value_and_grad(loss_fn)(state.params, x, y)
        grad_norm = _tree_l2_norm(grads)
        grads = _clip_grads(grads, grad_norm, grad_clip_norm)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        metrics = {
            "loss": loss_value,
            "grad_norm": grad_norm,
        }
        if compute_prediction_metrics:
            prediction = predict_fn(params, x)
            if metrics_fn is None:
                metrics["mse"] = jnp.mean((prediction - y) ** 2)
            else:
                metrics.update(metrics_fn(prediction, y))
        return (
            TrainState(
                params=params,
                optimizer=state.optimizer,
                opt_state=opt_state,
                step=state.step + jnp.asarray(1, dtype=jnp.int32),
            ),
            metrics,
        )

    return jax.jit(_step)


def _tree_l2_norm(tree: Params) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(jnp.asarray(leaf))) for leaf in leaves)
    )


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


def make_predict_fn(
    predict_fn: Callable[[Params, jnp.ndarray], jnp.ndarray],
):
    """Create a JIT-compiled prediction function for a fixed model config."""
    return jax.jit(predict_fn)


def train_epochs(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: Any,
    epochs: int,
    batch_size: int,
) -> tuple[TrainState, list[dict[str, float]]]:
    compiled_step = make_train_step(config, state.optimizer)
    history = []
    for _ in range(epochs):
        losses = []
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            state, metrics = compiled_step(
                state,
                x[start:stop],
                y[start:stop],
            )
            losses.append(metrics["loss"])
        history.append({"loss": float(jnp.mean(jnp.asarray(losses)))})
    return (
        replace(
            state,
            step=int(state.step),
        ),
        history,
    )


def train_epochs_with_step(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    compiled_step,
    epochs: int,
    batch_size: int,
) -> tuple[TrainState, list[dict[str, float]]]:
    history = []
    for _ in range(epochs):
        losses = []
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            state, metrics = compiled_step(
                state,
                x[start:stop],
                y[start:stop],
            )
            losses.append(metrics["loss"])
        history.append({"loss": float(jnp.mean(jnp.asarray(losses)))})
    return (
        replace(
            state,
            step=int(state.step),
        ),
        history,
    )
