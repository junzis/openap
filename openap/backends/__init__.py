"""OpenAP backends for different computational frameworks.

This module provides a unified interface for mathematical operations
across different backends:

- NumpyBackend: Default backend using NumPy (always available)
- CasadiBackend: Symbolic computation for optimization (requires casadi)
- JaxBackend: Autodiff and JIT compilation (requires jax)

Usage:
    from openap.backends import NumpyBackend, CasadiBackend, JaxBackend

    # Default NumPy backend
    thrust = openap.Thrust("A320")

    # CasADi backend for trajectory optimization
    thrust = openap.Thrust("A320", backend=CasadiBackend())

    # JAX backend for autodiff/JIT
    thrust = openap.Thrust("A320", backend=JaxBackend())
"""

from typing import Literal, Union

from .casadi_backend import CasadiBackend
from .jax_backend import JaxBackend
from .numpy_backend import NumpyBackend
from .protocol import MathBackend

# Type alias for backend selection
BackendType = Union[NumpyBackend, CasadiBackend, JaxBackend, MathBackend]
BackendName = Literal["numpy", "casadi", "jax"]

# Default backend instance (singleton)
_default_backend = NumpyBackend()


def get_backend(name: BackendName = "numpy") -> BackendType:
    """Get a backend instance by name.

    Args:
        name: Backend name ("numpy", "casadi", or "jax")

    Returns:
        Backend instance

    Raises:
        ValueError: If backend name is not recognized
    """
    if name == "numpy":
        return NumpyBackend()
    elif name == "casadi":
        return CasadiBackend()
    elif name == "jax":
        return JaxBackend()
    else:
        raise ValueError(
            f"Unknown backend: {name}. " "Choose from: 'numpy', 'casadi', 'jax'"
        )


def default_backend() -> NumpyBackend:
    """Get the default backend (NumPy).

    Returns:
        The default NumpyBackend instance
    """
    return _default_backend


__all__ = [
    "MathBackend",
    "NumpyBackend",
    "CasadiBackend",
    "JaxBackend",
    "BackendType",
    "BackendName",
    "get_backend",
    "default_backend",
]
