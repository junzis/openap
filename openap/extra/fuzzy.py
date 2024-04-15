"""
To prevent conda breaking, following functions are extracted 
from scikit-fuzzy library:

https://github.com/scikit-fuzzy/scikit-fuzzy
"""

import numpy as np


def gaussmf(x, mean, sigma):
    """
    Gaussian fuzzy membership function.

    Parameters
    ----------
    x : 1d array or iterable
        Independent variable.
    mean : float
        Gaussian parameter for center (mean) value.
    sigma : float
        Gaussian parameter for standard deviation.

    Returns
    -------
    y : 1d array
        Gaussian membership function for x.
    """
    return np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def zmf(x, a, b):
    """
    Z-function fuzzy membership generator.

    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : float
        'ceiling', where the function begins falling from 1.
    b : float
        'foot', where the function reattains zero.

    Returns
    -------
    y : 1d array
        Z-function.

    Notes
    -----
    Named such because of its Z-like shape.
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


def smf(x, a, b):
    """
    S-function fuzzy membership generator.

    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : float
        'foot', where the function begins to climb from zero.
    b : float
        'ceiling', where the function levels off at 1.

    Returns
    -------
    y : 1d array
        S-function.

    Notes
    -----
    Named such because of its S-like shape.
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


def interp_membership(x, xmf, xx, zero_outside_x=True):
    """
    Find the degree of membership ``u(xx)`` for a given value of ``x = xx``.

    Parameters
    ----------
    x : 1d array
        Independent discrete variable vector.
    xmf : 1d array
        Fuzzy membership function for ``x``.  Same length as ``x``.
    xx : float or array of floats
        Value(s) on universe ``x`` where the interpolated membership is
        desired.
    zero_outside_x : bool, optional
        Defines the behavior if ``xx`` contains value(s) which are outside the
        universe range as defined by ``x``.  If `True` (default), all
        extrapolated values will be zero.  If `False`, the first or last value
        in ``x`` will be what is returned to the left or right of the range,
        respectively.

    Returns
    -------
    xxmf : float or array of floats
        Membership function value at ``xx``, ``u(xx)``.  If ``xx`` is a single
        value, this will be a single value; if it is an array or iterable the
        result will be returned as a NumPy array of like shape.

    Notes
    -----
    For use in Fuzzy Logic, where an interpolated discrete membership function
    u(x) for discrete values of x on the universe of ``x`` is given. Then,
    consider a new value x = xx, which does not correspond to any discrete
    values of ``x``. This function computes the membership value ``u(xx)``
    corresponding to the value ``xx`` using linear interpolation.

    """
    if not zero_outside_x:
        kwargs = (None, None)
    else:
        kwargs = (0, 0)
    return np.interp(xx, x, xmf, left=kwargs[0], right=kwargs[1])


def defuzz(x, mfx, mode):
    """
    Defuzzification of a membership function, returning a defuzzified value
    of the function at x, using various defuzzification methods.

    Parameters
    ----------
    x : 1d array or iterable, length N
        Independent variable.
    mfx : 1d array of iterable, length N
        Fuzzy membership function.
    mode : string
        Controls which defuzzification method will be used.
        * 'mom'     : mean of maximum
        * 'som'     : min of maximum
        * 'lom'     : max of maximum

    Returns
    -------
    u : float or int
        Defuzzified result.

    Raises
    ------
    - EmptyMembershipError : When the membership function area is empty.
    - InconsistentMFDataError : When the length of the 'x' and the fuzzy
        membership function arrays are not equal.

    See Also
    --------
    skfuzzy.defuzzify.centroid, skfuzzy.defuzzify.dcentroid
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
