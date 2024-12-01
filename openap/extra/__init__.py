import functools

import numpy as np


def ndarrayconvert(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        new_args = []
        new_kwargs = {}

        for arg in args:
            if not isinstance(arg, str):
                arg = (
                    np.array([arg]).reshape(-1, 1)
                    if np.ndim(arg) == 0
                    else np.array(arg).reshape(-1, 1)
                )
            new_args.append(arg)

        for k, arg in kwargs.items():
            if not isinstance(arg, str):
                arg = (
                    np.array([arg]).reshape(-1, 1)
                    if np.ndim(arg) == 0
                    else np.array(arg).reshape(-1, 1)
                )
            new_kwargs[k] = arg

        result = func(self, *new_args, **new_kwargs)

        def scalar_convert(value):
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return value.item()
                if value.shape[0] == 1:
                    return value.item()
            return value

        if isinstance(result, tuple):
            return tuple(scalar_convert(r) for r in result)
        else:
            return scalar_convert(result)

        return result

    wrapper.orig_func = func
    return wrapper
