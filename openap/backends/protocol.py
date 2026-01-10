"""Backend protocol definition for OpenAP.

This module defines the interface that all math backends must implement.
Backends provide mathematical operations that can be swapped between
NumPy (default), CasADi (symbolic), and JAX (autodiff/JIT).
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MathBackend(Protocol):
    """Protocol defining math operations for different backends.

    All backends must provide these operations. The operations should
    work with the backend's native array types (numpy.ndarray, casadi.MX,
    jax.Array, etc.).
    """

    # --- Basic Math Functions ---

    def sqrt(self, x: Any) -> Any:
        """Square root."""
        ...

    def exp(self, x: Any) -> Any:
        """Exponential function."""
        ...

    def log(self, x: Any) -> Any:
        """Natural logarithm."""
        ...

    def power(self, x: Any, y: Any) -> Any:
        """Power function x^y."""
        ...

    # --- Trigonometric Functions ---

    def sin(self, x: Any) -> Any:
        """Sine function."""
        ...

    def cos(self, x: Any) -> Any:
        """Cosine function."""
        ...

    def tan(self, x: Any) -> Any:
        """Tangent function."""
        ...

    def arcsin(self, x: Any) -> Any:
        """Inverse sine function."""
        ...

    def arccos(self, x: Any) -> Any:
        """Inverse cosine function."""
        ...

    def arctan(self, x: Any) -> Any:
        """Inverse tangent function."""
        ...

    def arctan2(self, y: Any, x: Any) -> Any:
        """Two-argument inverse tangent."""
        ...

    # --- Comparison and Conditional ---

    def abs(self, x: Any) -> Any:
        """Absolute value."""
        ...

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Conditional selection: where(cond, x, y) = x if cond else y."""
        ...

    def maximum(self, x: Any, y: Any) -> Any:
        """Element-wise maximum."""
        ...

    def minimum(self, x: Any, y: Any) -> Any:
        """Element-wise minimum."""
        ...

    def clip(self, x: Any, min_val: Any, max_val: Any) -> Any:
        """Clip values to range [min_val, max_val]."""
        ...

    # --- Interpolation ---

    def interp(self, x: Any, xp: Any, fp: Any) -> Any:
        """Linear interpolation."""
        ...

    # --- Array Creation ---

    def linspace(self, start: Any, stop: Any, num: int) -> Any:
        """Create evenly spaced values."""
        ...

    # --- Modulo ---

    def fmod(self, x: Any, y: Any) -> Any:
        """Floating-point modulo operation."""
        ...

    # --- Constants ---

    @property
    def pi(self) -> float:
        """The constant pi."""
        ...
