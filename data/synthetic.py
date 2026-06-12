from __future__ import annotations

import jax
import jax.numpy as jnp


def _minmax_scale(values: jnp.ndarray) -> jnp.ndarray:
    lo = jnp.min(values, axis=0, keepdims=True)
    hi = jnp.max(values, axis=0, keepdims=True)
    denom = jnp.where(hi - lo < 1e-8, 1.0, hi - lo)
    return 2.0 * (values - lo) / denom - 1.0


def make_mackey_glass(
    length: int,
    tau: int = 35,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    dt: float = 1.0,
    seed: int = 0,
) -> jnp.ndarray:
    """Generate a normalized Mackey-Glass sequence with Euler integration."""
    del seed
    if length <= 0:
        raise ValueError("length must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")

    history = [1.2 + 0.01 * jnp.sin(i) for i in range(tau + 1)]
    for t in range(length - 1):
        x_t = history[-1]
        x_tau = history[-tau]
        dx = beta * x_tau / (1.0 + x_tau**n) - gamma * x_t
        history.append(x_t + dt * dx)

    series = jnp.asarray(history[-length:], dtype=jnp.float32).reshape(length, 1)
    return _minmax_scale(series)


def make_henon_map(
    length: int,
    a: float = 1.4,
    b: float = 0.3,
    initial_state: tuple[float, float] = (0.0, 0.0),
) -> jnp.ndarray:
    """Generate a normalized two-feature Henon map trajectory."""
    if length <= 0:
        raise ValueError("length must be positive")
    x, y = initial_state
    values = []
    for _ in range(length):
        x, y = 1.0 - a * x * x + y, b * x
        values.append((x, y))
    return _minmax_scale(jnp.asarray(values, dtype=jnp.float32))


def make_lorenz_system(
    length: int,
    rho: float = 28.0,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    initial_state: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> jnp.ndarray:
    """Generate a normalized Lorenz trajectory with Euler integration."""
    if length <= 0:
        raise ValueError("length must be positive")
    x, y, z = initial_state
    values = []
    for _ in range(length):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x, y, z = x + dt * dx, y + dt * dy, z + dt * dz
        values.append((x, y, z))
    return _minmax_scale(jnp.asarray(values, dtype=jnp.float32))


def make_narma(
    length: int,
    order: int = 10,
    seed: int = 0,
    input_scale: float = 0.5,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 1.5,
    delta: float = 0.1,
) -> jnp.ndarray:
    """Generate a NARMA control-target sequence.

    Returns two columns: the external input ``u_t`` and the NARMA target
    ``y_t``. The common benchmark setting samples ``u_t`` from [0, 0.5].
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if order <= 0:
        raise ValueError("order must be positive")
    if length <= order + 1:
        raise ValueError("length must exceed order + 1")

    key = jax.random.PRNGKey(seed)
    u = jax.random.uniform(
        key,
        (length + order + 1,),
        minval=0.0,
        maxval=input_scale,
        dtype=jnp.float32,
    )
    y = [jnp.asarray(0.0, dtype=jnp.float32) for _ in range(order + 1)]
    for t in range(order, length + order):
        recent = jnp.stack(y[t - order + 1 : t + 1])
        next_y = (
            alpha * y[t]
            + beta * y[t] * jnp.sum(recent)
            + gamma * u[t - order + 1] * u[t]
            + delta
        )
        y.append(next_y)

    targets = jnp.asarray(y[order + 1 : order + 1 + length], dtype=jnp.float32)
    inputs = u[order + 1 : order + 1 + length]
    return jnp.stack([inputs, targets], axis=-1).astype(jnp.float32)
