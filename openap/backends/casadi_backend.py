"""CasADi backend for OpenAP.

This backend provides symbolic computation support using CasADi,
enabling automatic differentiation and optimization.
"""

from typing import Any


class CasadiBackend:
    """CasADi backend implementation.

    This backend uses CasADi for symbolic computations, enabling:
    - Symbolic differentiation
    - Integration with NLP solvers (IPOPT, etc.)
    - Trajectory optimization (openap-top)

    CasADi is imported lazily to avoid requiring it as a dependency
    for users who only need NumPy functionality.
    """

    def __init__(self):
        """Initialize CasADi backend with lazy import."""
        self._ca = None

    @property
    def ca(self):
        """Lazy import of CasADi."""
        if self._ca is None:
            try:
                import casadi

                self._ca = casadi
            except ImportError:
                raise ImportError(
                    "CasADi is required for symbolic computations. "
                    "Install with: pip install casadi"
                )
        return self._ca

    # --- Basic Math Functions ---

    def sqrt(self, x: Any) -> Any:
        return self.ca.sqrt(x)

    def exp(self, x: Any) -> Any:
        return self.ca.exp(x)

    def log(self, x: Any) -> Any:
        return self.ca.log(x)

    def power(self, x: Any, y: Any) -> Any:
        return self.ca.power(x, y)

    # --- Trigonometric Functions ---

    def sin(self, x: Any) -> Any:
        return self.ca.sin(x)

    def cos(self, x: Any) -> Any:
        return self.ca.cos(x)

    def tan(self, x: Any) -> Any:
        return self.ca.tan(x)

    def arcsin(self, x: Any) -> Any:
        return self.ca.asin(x)

    def arccos(self, x: Any) -> Any:
        return self.ca.acos(x)

    def arctan(self, x: Any) -> Any:
        return self.ca.atan(x)

    def arctan2(self, y: Any, x: Any) -> Any:
        return self.ca.atan2(y, x)

    # --- Comparison and Conditional ---

    def abs(self, x: Any) -> Any:
        return self.ca.fabs(x)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return self.ca.if_else(condition, x, y)

    def maximum(self, x: Any, y: Any) -> Any:
        return self.ca.fmax(x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        return self.ca.fmin(x, y)

    def clip(self, x: Any, min_val: Any, max_val: Any) -> Any:
        return self.ca.fmax(min_val, self.ca.fmin(x, max_val))

    # --- Interpolation ---

    def interp(self, x: Any, xp: Any, fp: Any) -> Any:
        """Linear interpolation using CasADi interpolant.

        Note: xp and fp must be Python lists or tuples, not symbolic.
        """
        lut = self.ca.interpolant("LUT", "linear", [list(xp)], list(fp))
        return lut(x)

    # --- Array Creation ---

    def linspace(self, start: Any, stop: Any, num: int) -> Any:
        """Create evenly spaced values using NumPy.

        Note: This returns a NumPy array since CasADi doesn't have
        a native linspace. Typically used for plotting, not symbolic ops.
        """
        import numpy as np

        return np.linspace(start, stop, num)

    # --- Constants ---

    @property
    def pi(self) -> float:
        return 3.141592653589793
