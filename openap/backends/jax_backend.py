"""JAX backend for OpenAP.

This backend provides JAX support for automatic differentiation,
JIT compilation, and GPU acceleration.
"""

from typing import Any


class JaxBackend:
    """JAX backend implementation.

    This backend uses JAX for:
    - Automatic differentiation (grad, jacobian, hessian)
    - JIT compilation for performance
    - GPU/TPU acceleration
    - Vectorization (vmap)

    JAX is imported lazily to avoid requiring it as a dependency
    for users who only need NumPy functionality.
    """

    def __init__(self):
        """Initialize JAX backend with lazy import."""
        self._jnp = None

    @property
    def jnp(self):
        """Lazy import of JAX numpy."""
        if self._jnp is None:
            try:
                import jax.numpy as jnp

                self._jnp = jnp
            except ImportError:
                raise ImportError(
                    "JAX is required for this backend. "
                    "Install with: pip install jax jaxlib"
                )
        return self._jnp

    # --- Basic Math Functions ---

    def sqrt(self, x: Any) -> Any:
        return self.jnp.sqrt(x)

    def exp(self, x: Any) -> Any:
        return self.jnp.exp(x)

    def log(self, x: Any) -> Any:
        return self.jnp.log(x)

    def power(self, x: Any, y: Any) -> Any:
        return self.jnp.power(x, y)

    # --- Trigonometric Functions ---

    def sin(self, x: Any) -> Any:
        return self.jnp.sin(x)

    def cos(self, x: Any) -> Any:
        return self.jnp.cos(x)

    def tan(self, x: Any) -> Any:
        return self.jnp.tan(x)

    def arcsin(self, x: Any) -> Any:
        return self.jnp.arcsin(x)

    def arccos(self, x: Any) -> Any:
        return self.jnp.arccos(x)

    def arctan(self, x: Any) -> Any:
        return self.jnp.arctan(x)

    def arctan2(self, y: Any, x: Any) -> Any:
        return self.jnp.arctan2(y, x)

    # --- Comparison and Conditional ---

    def abs(self, x: Any) -> Any:
        return self.jnp.abs(x)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return self.jnp.where(condition, x, y)

    def maximum(self, x: Any, y: Any) -> Any:
        return self.jnp.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        return self.jnp.minimum(x, y)

    def clip(self, x: Any, min_val: Any, max_val: Any) -> Any:
        return self.jnp.clip(x, min_val, max_val)

    # --- Interpolation ---

    def interp(self, x: Any, xp: Any, fp: Any) -> Any:
        """Linear interpolation.

        Note: xp and fp are converted to JAX arrays if they are lists.
        """
        xp = self.jnp.array(xp) if isinstance(xp, list) else xp
        fp = self.jnp.array(fp) if isinstance(fp, list) else fp
        return self.jnp.interp(x, xp, fp)

    # --- Array Creation ---

    def linspace(self, start: Any, stop: Any, num: int) -> Any:
        return self.jnp.linspace(start, stop, num)

    # --- Constants ---

    @property
    def pi(self) -> float:
        return self.jnp.pi
