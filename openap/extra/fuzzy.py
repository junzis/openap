"""Fuzzy logic membership functions.

Extracted from scikit-fuzzy library to prevent conda breaking:
https://github.com/scikit-fuzzy/scikit-fuzzy
"""

import numpy as np


def gaussmf(x, mean: float, sigma: float):
    """Gaussian fuzzy membership function.

    Args:
        x: Independent variable (1d array or iterable).
        mean: Gaussian parameter for center (mean) value.
        sigma: Gaussian parameter for standard deviation.

    Returns:
        Gaussian membership function for x.

    """
    return np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def zmf(x, a: float, b: float):
    """Z-function fuzzy membership generator.

    Named for its Z-like shape.

    Args:
        x: Independent variable (1d array).
        a: Ceiling, where the function begins falling from 1.
        b: Foot, where the function reattains zero.

    Returns:
        Z-function membership values.

    """
    assert a <= b, "a <= b is required."

    y = np.ones(len(x))

    idx = np.logical_and(a <= x, x < (a + b) / 2)
    y[idx] = 1 - 2 * ((x[idx] - a) / (b - a)) ** 2

    idx = np.logical_and((a + b) / 2 <= x, x <= b)
    y[idx] = 2 * ((x[idx] - b) / (b - a)) ** 2

    idx = x >= b
    y[idx] = 0

    return y


def smf(x, a: float, b: float):
    """S-function fuzzy membership generator.

    Named for its S-like shape.

    Args:
        x: Independent variable (1d array).
        a: Foot, where the function begins to climb from zero.
        b: Ceiling, where the function levels off at 1.

    Returns:
        S-function membership values.

    """
    assert a <= b, "a <= b is required."

    y = np.ones(len(x))
    idx = x <= a
    y[idx] = 0

    idx = np.logical_and(a <= x, x <= (a + b) / 2)
    y[idx] = 2 * ((x[idx] - a) / (b - a)) ** 2

    idx = np.logical_and((a + b) / 2 <= x, x <= b)
    y[idx] = 1 - 2 * ((x[idx] - b) / (b - a)) ** 2

    return y


def interp_membership(x, xmf, xx, zero_outside_x: bool = True):
    """Find the degree of membership u(xx) for a given value of x = xx.

    For use in Fuzzy Logic, where an interpolated discrete membership function
    u(x) for discrete values of x on the universe of x is given. This function
    computes the membership value u(xx) using linear interpolation.

    Args:
        x: Independent discrete variable vector (1d array).
        xmf: Fuzzy membership function for x. Same length as x.
        xx: Value(s) on universe x where the interpolated membership is desired.
        zero_outside_x: If True, extrapolated values will be zero. If False,
            the first or last value in x will be returned. Defaults to True.

    Returns:
        Membership function value at xx, u(xx).

    """
    if not zero_outside_x:
        kwargs = (None, None)
    else:
        kwargs = (0, 0)
    return np.interp(xx, x, xmf, left=kwargs[0], right=kwargs[1])


def defuzz(x, mfx, mode: str):
    """Defuzzification of a membership function.

    Args:
        x: Independent variable (1d array, length N).
        mfx: Fuzzy membership function (1d array, length N).
        mode: Defuzzification method ('mom': mean of maximum,
            'som': min of maximum, 'lom': max of maximum).

    Returns:
        Defuzzified result.

    Raises:
        ValueError: When membership function data is inconsistent or mode invalid.

    """
    mode = mode.lower()
    x = x.ravel()
    mfx = mfx.ravel()
    n = len(x)
    if n != len(mfx):
        raise ValueError("inconsistent membership function")

    elif "mom" in mode:
        return np.mean(x[mfx == mfx.max()])

    elif "som" in mode:
        return np.min(x[mfx == mfx.max()])

    elif "lom" in mode:
        return np.max(x[mfx == mfx.max()])

    else:
        raise ValueError(f"The mode: {mode} was incorrect.")
