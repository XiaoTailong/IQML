from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import jax
import tensorcircuit as tc


def configure_tensorcircuit_jax(
    *,
    require_ng: bool = False,
    enable_x64: bool = True,
    dtype: str = "complex128",
) -> None:
    """Configure TensorCircuit through its ``tensorcircuit`` import name."""
    if require_ng:
        try:
            version("tensorcircuit-ng")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "This project requires TensorCircuit. Install it with "
                "`python -m pip install tensorcircuit` or, if available for your "
                "environment, `python -m pip install tensorcircuit-ng`. The import "
                "name remains `tensorcircuit`."
            ) from exc
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype(dtype)
    backend_name = getattr(tc.backend, "name", "")
    if backend_name != "jax":
        raise RuntimeError(f"TensorCircuit backend must be jax, got {backend_name!r}")


def tensorcircuit_runtime_info() -> dict[str, str]:
    try:
        tc_ng_version = version("tensorcircuit-ng")
    except PackageNotFoundError:
        tc_ng_version = "not-installed"
    try:
        legacy_tc_version = version("tensorcircuit")
    except PackageNotFoundError:
        legacy_tc_version = "not-installed"
    return {
        "tensorcircuit_module": str(getattr(tc, "__file__", "unknown")),
        "tensorcircuit_module_version": str(getattr(tc, "__version__", "unknown")),
        "tensorcircuit_ng_distribution": tc_ng_version,
        "tensorcircuit_distribution": legacy_tc_version,
        "tensorcircuit_backend": str(getattr(tc.backend, "name", "unknown")),
        "tensorcircuit_dtype": str(getattr(tc, "dtypestr", "unknown")),
        "tensorcircuit_real_dtype": str(getattr(tc, "rdtypestr", "unknown")),
        "jax_enable_x64": str(bool(jax.config.jax_enable_x64)),
    }
