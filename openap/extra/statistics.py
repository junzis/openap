"""Fit data using different statistical models."""

import scipy.stats
from matplotlib import pyplot as plt

import numpy as np


def fit(data, models):
    if not isinstance(models, list):
        if isinstance(models, str):
            models = [models]
        else:
            raise RuntimeError("models must be string or list of strings.")

    data = np.array(data)
    data = data[np.isfinite(data)]
    data = data[data > np.percentile(data, 0.025)]
    data = data[data < np.percentile(data, 99.975)]

    # split data in training and testing 50-50
    train, test = data[0:-1:1], data[1:-1:1]

    # construct the bound for the model fitting
    dmin = min(data) - 1e-8
    dmax = max(data) + 1e-8
    dscale = dmax - dmin

    result = dict()
    for model in models:
        # fit distribution and run kstest
        dist = getattr(scipy.stats, model)
        if model == "norm":
            param_train = dist.fit(train)
        elif model == "gamma":
            param_train = dist.fit(train, floc=dmin)
        else:
            param_train = dist.fit(train, floc=dmin, fscale=dscale)

        ks = scipy.stats.kstest(test, model, param_train)
        error = ks[0]  # D-stats

        # recompute distribution based on all data
        if model == "norm":
            param = dist.fit(data)
        elif model == "gamma":
            param = dist.fit(data, floc=dmin)
        else:
            param = dist.fit(data, floc=dmin, fscale=dscale)

        # construct the example PDF for ploting
        ci = dist.interval(0.999, *param)
        xmin = ci[0] - dscale * 0.05
        xmax = ci[1] + dscale * 0.05
        pdfx = np.linspace(xmin, xmax, 1000)
        pdfy = dist.pdf(pdfx, *param)

        result[model] = dict()
        result[model]["param"] = param
        result[model]["pdfx"] = pdfx
        result[model]["pdfy"] = pdfy
        result[model]["error"] = error

        # print(model, param, error)

    return result


def fitplot(data, model, **kwargs):
    fitresults = fit(data, model)

    if "bins" in kwargs:
        bins = kwargs["bins"]
        del kwargs["bins"]
    else:
        bins = 20

    data = np.array(data)
    data = data[np.isfinite(data)]
    plt.hist(data, bins=bins, normed=True, color="gray", edgecolor="none", alpha=0.3)
    plt.plot(
        fitresults[model]["pdfx"], fitresults[model]["pdfy"], label=model, **kwargs
    )
    # plt.legend(loc='best')
    plt.xlim([min(data), max(data)])
    plt.ylabel("density (-)")
    plt.grid()
    return plt
