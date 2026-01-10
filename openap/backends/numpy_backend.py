"""NumPy backend for OpenAP.

This is the default backend using NumPy for numerical computations.
"""

from typing import Any

import numpy as np


class NumpyBackend:
    """NumPy backend implementation.

    This is the default backend for OpenAP, providing numerical
    computations using NumPy arrays.
    """

    # --- Basic Math Functions ---

    @staticmethod
    def sqrt(x: Any) -> Any:
        return np.sqrt(x)

    @staticmethod
    def exp(x: Any) -> Any:
        return np.exp(x)

    @staticmethod
    def log(x: Any) -> Any:
        return np.log(x)

    @staticmethod
    def power(x: Any, y: Any) -> Any:
        return np.power(x, y)

    # --- Trigonometric Functions ---

    @staticmethod
    def sin(x: Any) -> Any:
        return np.sin(x)

    @staticmethod
    def cos(x: Any) -> Any:
        return np.cos(x)

    @staticmethod
    def tan(x: Any) -> Any:
        return np.tan(x)

    @staticmethod
    def arcsin(x: Any) -> Any:
        return np.arcsin(x)

    @staticmethod
    def arccos(x: Any) -> Any:
        return np.arccos(x)

    @staticmethod
    def arctan(x: Any) -> Any:
        return np.arctan(x)

    @staticmethod
    def arctan2(y: Any, x: Any) -> Any:
        return np.arctan2(y, x)

    # --- Comparison and Conditional ---

    @staticmethod
    def abs(x: Any) -> Any:
        return np.abs(x)

    @staticmethod
    def where(condition: Any, x: Any, y: Any) -> Any:
        return np.where(condition, x, y)

    @staticmethod
    def maximum(x: Any, y: Any) -> Any:
        return np.maximum(x, y)

    @staticmethod
    def minimum(x: Any, y: Any) -> Any:
        return np.minimum(x, y)

    @staticmethod
    def clip(x: Any, min_val: Any, max_val: Any) -> Any:
        return np.clip(x, min_val, max_val)

    # --- Interpolation ---

    @staticmethod
    def interp(x: Any, xp: Any, fp: Any) -> Any:
        return np.interp(x, xp, fp)

    # --- Array Creation ---

    @staticmethod
    def linspace(start: Any, stop: Any, num: int) -> Any:
        return np.linspace(start, stop, num)

    # --- Modulo ---

    @staticmethod
    def fmod(x: Any, y: Any) -> Any:
        return np.fmod(x, y)

    # --- Constants ---

    @property
    def pi(self) -> float:
        return np.pi
