import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import filters
from scipy.signal.windows import gaussian


class BaseFilter(object):
    def __init__(self, i=False):
        self.interpolate = i

    def sortxy(self, X, Y):
        XY = list(zip(X, Y))
        XY.sort(key=lambda t: t[0])
        X1, Y1 = list(zip(*XY))
        return np.array(X1), np.array(Y1)

    def simplefill(self, X, Y):
        """Fill the missing data with closest previous data each second"""

        X, Y = self.sortxy(X, Y)

        X = np.array(X)
        Y = np.array(Y)

        Xfull = list(range(int(X[0]), int(X[-1] + 1)))
        Yfull = []

        y = 0
        for x in Xfull:
            try:
                i = np.where(X == x)[0][0]
                y = Y[i]
            except:
                pass

            Yfull.append(y)

        return np.array(Xfull), np.array(Yfull)

    def filterplot(self, x, y, xf, yf):
        plt.plot(x, y, ".", color="blue", alpha=0.5)
        plt.plot(xf, yf, "-", color="red")


class SavitzkyGolay(BaseFilter):
    """
    SavitzkyGolay Filter

    Parameters
    ----------
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0
        means only smoothing)

    """

    def __init__(self, window_size=11, order=2, deriv=0, i=False):
        super(SavitzkyGolay, self).__init__(i=i)

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("win size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("win size is too small for the polynomials order")

        self.window_size = window_size
        self.order = order
        self.deriv = deriv

    def filter(self, X, Y):
        if self.interpolate:
            X, Y = self.simplefill(X, Y)
        else:
            X, Y = self.sortxy(X, Y)

        order_range = list(range(self.order + 1))
        half_window = (self.window_size - 1) // 2
        # precompute coefficients
        b = np.mat(
            [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
        )
        m = np.linalg.pinv(b).A[self.deriv]
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = Y[0] - np.abs(Y[1 : half_window + 1][::-1] - Y[0])
        lastvals = Y[-1] + np.abs(Y[-half_window - 1 : -1][::-1] - Y[-1])
        Y1 = np.concatenate((firstvals, Y, lastvals))
        Y2 = np.convolve(m, Y1, mode="valid")

        return X, Y2


class Spline(BaseFilter):
    """
    Spline smoothing

    """

    def __init__(self, k=1, i=False):
        super(Spline, self).__init__(i=i)
        self.k = k

    def kernel(self, series, sigma=3):
        # fix the weight of data
        # http://www.nehalemlabs.net/prototype/blog/2014/04/12/
        #    how-to-fix-scipys-interpolating-spline-default-behavior/
        series = np.asarray(series)
        b = gaussian(25, sigma)
        averages = filters.convolve1d(series, b / b.sum())
        variances = filters.convolve1d(np.power(series - averages, 2), b / b.sum())
        variances[variances == 0] = 1
        return averages, variances

    def filter(self, X, Y):
        X, Y = self.sortxy(X, Y)

        # using gaussian kernel to get a better variances
        avg, var = self.kernel(Y)
        spl = UnivariateSpline(X, Y, k=self.k, w=1 / np.sqrt(var))

        if self.interpolate:
            xmax = X[-1]
            Xfull = np.arange(xmax)
            Yfull = spl(Xfull)
            return Xfull, Yfull
        else:
            Y1 = spl(X)
            return X, Y1


class TWF(BaseFilter):
    """
    Time-based weighted filter
    input X is the time stamps of Y
    """

    def __init__(self, window_size=10):
        super(TWF, self).__init__()
        self.window_size = window_size

    def filter(self, X, Y):
        X, Y = self.sortxy(X, Y)

        YF = np.zeros(Y.shape)
        YF[0] = Y[0]
        YF[1] = Y[1]

        for i in range(2, len(X)):
            if i < self.window_size:
                y = (np.average(YF[: i - 1]) + Y[i]) / 2.0
            else:
                Xwin = X[i - self.window_size : i - 1][::-1]
                Ywin = YF[i - self.window_size : i - 1][::-1]
                dXwin = Xwin[1:] - Xwin[:-1]
                yw = (Ywin[0] + np.sum(1.0 / dXwin * Ywin[1:])) / (
                    1 + np.sum(1.0 / dXwin)
                )
                y = (yw + Y[i]) / 2.0
            YF[i] = y
        return X, np.array(YF)
