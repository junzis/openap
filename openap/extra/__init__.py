"""Extra utilities for OpenAP."""

import functools

import numpy as np


def _is_numpy_compatible(arg):
    """Check if argument should be converted to NumPy array.

    Returns False for CasADi and JAX types to avoid conversion.
    """
    if arg is None:
        return True

    # Check for CasADi types
    try:
        import casadi

        if isinstance(arg, (casadi.SX, casadi.MX, casadi.DM)):
            return False
    except ImportError:
        pass

    # Check for JAX types
    try:
        import jax

        if isinstance(arg, jax.Array):
            return False
    except ImportError:
        pass

    return True


def ndarrayconvert(func=None, column=False):
    """Decorator to convert inputs to NumPy arrays and handle scalar outputs.

    This decorator:
    - Converts scalar inputs to 1-element arrays
    - Converts list inputs to arrays
    - Converts 1-element array outputs back to scalars

    For non-NumPy backends (CasADi, JAX), the decorator passes through
    without conversion to allow symbolic computation and autodiff.

    Args:
        func: Function to decorate.
        column: If True, reshape arrays to column vectors.
    """
    assert func is None or callable(func)

    def _decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if we're using a non-NumPy backend
            backend = getattr(self, "backend", None)
            use_numpy = backend is None or type(backend).__name__ == "NumpyBackend"

            # Also check if any argument is a non-NumPy type
            all_numpy = all(_is_numpy_compatible(arg) for arg in args)
            all_numpy = all_numpy and all(
                _is_numpy_compatible(v) for v in kwargs.values()
            )

            # If using non-NumPy backend or non-NumPy arguments, skip conversion
            if not use_numpy or not all_numpy:
                return func(self, *args, **kwargs)

            # NumPy path: convert inputs
            new_args = []
            new_kwargs = {}

            for arg in args:
                if not isinstance(arg, str):
                    if np.ndim(arg) == 0:
                        arg = np.array([arg])
                    else:
                        arg = np.array(arg)

                    if column:
                        arg = arg.reshape(-1, 1)
                new_args.append(arg)

            for k, arg in kwargs.items():
                if not isinstance(arg, str):
                    if np.ndim(arg) == 0:
                        arg = np.array([arg])
                    else:
                        arg = np.array(arg)

                    if column:
                        arg = arg.reshape(-1, 1)
                new_kwargs[k] = arg

            result = func(self, *new_args, **new_kwargs)

            def scalar_convert(value):
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        return value.item()
                    elif value.ndim == 1 and value.shape[0] == 1:
                        return value.item()
                    elif (
                        not column
                        and value.ndim > 1
                        and (value.shape[0] == 1 or value.shape[1] == 1)
                    ):
                        return value.squeeze()
                return value

            if isinstance(result, tuple):
                return tuple(scalar_convert(r) for r in result)
            else:
                return scalar_convert(result)

        wrapper.orig_func = func
        return wrapper

    return _decorator(func) if callable(func) else _decorator
