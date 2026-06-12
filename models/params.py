from __future__ import annotations

from typing import Any

import jax


def count_parameters(params: Any) -> int:
    """Count scalar trainable parameters in a pytree."""
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves))
