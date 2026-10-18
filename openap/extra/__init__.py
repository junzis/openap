"""Extra utilities for OpenAP."""

import functools

import numpy as np


def _is_symbolic_type(arg):
    """Check if argument is a symbolic type (CasADi or JAX).

    Returns True for CasADi SX/MX/DM and JAX Array types.
    """
    if arg is None:
        return False

    type_name = type(arg).__module__

    # Fast path: check module name prefix
    if type_name.startswith("casadi"):
        return True
    if type_name.startswith("jax"):
        return True

    return False


def ndarrayconvert(func=None, column=False):
    """Decorator to convert inputs to NumPy arrays and handle scalar outputs.

    This decorator:
    - Converts scalar inputs to 1-element arrays
    - Converts list inputs to arrays
    - Converts 1-element array outputs back to scalars

    For symbolic types (CasADi, JAX), the decorator passes through
    without conversion to allow symbolic computation and autodiff.

    Args:
        func: Function to decorate.
        column: If True, reshape arrays to column vectors.
    """

    def _decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # If any argument is symbolic (CasADi/JAX), skip conversion
            if any(_is_symbolic_type(arg) for arg in args):
                return func(self, *args, **kwargs)
            if any(_is_symbolic_type(v) for v in kwargs.values()):
                return func(self, *args, **kwargs)

            # NumPy path: convert inputs to arrays
            new_args = []
            for arg in args:
                if isinstance(arg, str):
                    new_args.append(arg)
                elif np.ndim(arg) == 0:
                    arr = np.array([arg])
                    new_args.append(arr.reshape(-1, 1) if column else arr)
                else:
                    arr = np.asarray(arg)
                    new_args.append(arr.reshape(-1, 1) if column else arr)

            new_kwargs = {}
            for k, arg in kwargs.items():
                if isinstance(arg, str):
                    new_kwargs[k] = arg
                elif np.ndim(arg) == 0:
                    arr = np.array([arg])
                    new_kwargs[k] = arr.reshape(-1, 1) if column else arr
                else:
                    arr = np.asarray(arg)
                    new_kwargs[k] = arr.reshape(-1, 1) if column else arr

            result = func(self, *new_args, **new_kwargs)

            # Convert 1-element arrays back to scalars
            def to_scalar(value):
                if not isinstance(value, np.ndarray):
                    return value
                if value.size == 1:
                    return value.item()
                if not column and value.ndim > 1:
                    return value.squeeze()
                return value

            if isinstance(result, tuple):
                return tuple(to_scalar(r) for r in result)
            return to_scalar(result)

        wrapper.orig_func = func
        return wrapper

    if func is not None:
        return _decorator(func)
    return _decorator


def mass_to_kg(mass, mtow, backend=None):
    """Convert mass in kg or a fraction of MTOW to kg, element by element.

    Values in (0, 1] are MTOW fractions. Larger values are kilograms.
    Concrete inputs must be finite, positive, and no greater than MTOW.
    Symbolic/traced inputs retain their backend; their caller must enforce
    the same domain when evaluating or optimizing the resulting expression.
    Fuel integration must use kg internally, including the initial state.
    """
    from openap.backends import default_backend

    if not np.isfinite(mtow) or mtow <= 1:
        raise ValueError("MTOW must be finite and greater than 1 kg")
    if type(mass).__module__.startswith("casadi") and not mass.is_constant():
        values = None
    elif _is_symbolic_type(mass):
        try:
            values = np.asarray(mass, dtype=float)
        except (TypeError, ValueError, RuntimeError):
            values = None
    else:
        values = np.asarray(mass, dtype=float)
        mass = values
    if values is not None:
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("mass must be finite and positive")
        if np.any(values > mtow):
            raise ValueError(f"mass must not exceed MTOW ({mtow:g} kg)")
    return (backend or default_backend()).where(mass <= 1, mass * mtow, mass)
