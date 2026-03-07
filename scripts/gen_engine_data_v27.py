import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def clean_name(name):
    if "SelectOne" in name:
        name = name.split(" ")[0].strip()
    elif "Build Spec" in name:
        name = name.split(" ")[0].strip()
    elif "Block" in name:
        name = name.split(" ")[0].strip()
    elif "series" in name:
        name = name.split(" ")[0].strip()
    elif "(" in name:
        name = name.split("(")[0].strip()
    return name


def clean_bpr(bpr):
    try:
        bpr = float(bpr)
    except (ValueError, TypeError):
        bpr = "-"
    return bpr


f = "input/edb-emissions-databank v27 (web).xlsx"

xl = pd.ExcelFile(f)

df0 = xl.parse(sheet_name=2)

df0.dropna(subset=["UID"], inplace=True)

df = df0.iloc[
    1:,
    [
        0,
        1,
        3,
        4,
        5,
        6,
        11,
        12,
        17,
        19,
        22,
        23,
        24,
        25,
        35,
        36,
        37,
        38,
        48,
        49,
        50,
        51,
        80,
        81,
        82,
        83,
        84,
        94,
    ],
]
df.columns = [
    "uid",
    "name",
    "type",
    "bpr",
    "pr",
    "max_thrust",
    "superseded",
    "superseded_by",
    "out_production",
    "out_service",
    "ei_hc_to",
    "ei_hc_co",
    "ei_hc_app",
    "ei_hc_idl",
    "ei_co_to",
    "ei_co_co",
    "ei_co_app",
    "ei_co_idl",
    "ei_nox_to",
    "ei_nox_co",
    "ei_nox_app",
    "ei_nox_idl",
    "ff_to",
    "ff_co",
    "ff_app",
    "ff_idl",
    "fuel_lto",
    "manufacturer",
]

df.dropna(subset=["name"], inplace=True)

df.loc[:, "uid"] = df["uid"].str.strip()
df.loc[:, "name"] = df["name"].str.strip()
df.loc[:, "name"] = df["name"].apply(clean_name)

df.loc[:, "bpr"] = df["bpr"].apply(clean_bpr)

df.loc[:, "max_thrust"] = (df["max_thrust"] * 1000).astype(int)

df = df[df["superseded"] != "x"]
df = df[df["uid"] != ""]
df = df[df["bpr"] != "-"]
df = df[df["ei_hc_to"] != "-"]
df = df[df["ei_hc_co"] != "-"]
df = df[df["ei_hc_app"] != "-"]
df = df[df["ei_hc_idl"] != "-"]
df = df[df["ei_hc_to"] != "*"]
df = df[df["ei_hc_co"] != "*"]
df = df[df["ei_hc_app"] != "*"]
df = df[df["ei_hc_idl"] != "*"]
df = df[df["ei_co_to"] != "-"]
df = df[df["ei_co_co"] != "-"]
df = df[df["ei_co_app"] != "-"]
df = df[df["ei_co_idl"] != "-"]
df = df[df["ei_co_to"] != "*"]
df = df[df["ei_co_co"] != "*"]
df = df[df["ei_co_app"] != "*"]
df = df[df["ei_co_idl"] != "*"]
df = df[df["ei_nox_to"] != "-"]
df = df[df["ei_nox_co"] != "-"]
df = df[df["ei_nox_app"] != "-"]
df = df[df["ei_nox_idl"] != "-"]
df = df[df["ei_nox_to"] != "*"]
df = df[df["ei_nox_co"] != "*"]
df = df[df["ei_nox_app"] != "*"]
df = df[df["ei_nox_idl"] != "*"]
df = df[df["ff_to"] != "-"]
df = df[df["ff_co"] != "-"]
df = df[df["ff_app"] != "-"]
df = df[df["ff_idl"] != "-"]
df = df[df["fuel_lto"] != "-"]

df.drop_duplicates(subset=["name"], keep="last", inplace=True)

# Die supersededs vinden we nu niet meer interessant
df = df.loc[
    :,
    [
        "uid",
        "name",
        "manufacturer",
        "type",
        "bpr",
        "pr",
        "max_thrust",
        "ei_hc_to",
        "ei_hc_co",
        "ei_hc_app",
        "ei_hc_idl",
        "ei_co_to",
        "ei_co_co",
        "ei_co_app",
        "ei_co_idl",
        "ei_nox_to",
        "ei_nox_co",
        "ei_nox_app",
        "ei_nox_idl",
        "ff_to",
        "ff_co",
        "ff_app",
        "ff_idl",
        "fuel_lto",
    ],
]


def func_fuel3(x, c3, c2, c1, c0):
    return c3 * x ** 3 + c2 * x ** 2 + c1 * x + c0


def func_fuel2(x, a, b):
    return a * (x + b) ** 2


# def func_fuel(x, c1, c2):
#     return c1 * np.exp(c2 * x)


# compute fuel flow coefficient
x = [0.07, 0.3, 0.85, 1.0]
for i, r in df.iterrows():
    y = [r["ff_idl"], r["ff_app"], r["ff_co"], r["ff_to"]]

    # coef = np.polyfit(x, y, 2)
    # df.loc[i, 'fuel_c2'] = coef[0]
    # df.loc[i, 'fuel_c1'] = coef[1]
    # df.loc[i, 'fuel_c0'] = coef[2]

    coef, cov = curve_fit(func_fuel3, x, y)
    df.loc[i, "fuel_c3"] = coef[0]
    df.loc[i, "fuel_c2"] = coef[1]
    df.loc[i, "fuel_c1"] = coef[2]

    coef, cov = curve_fit(func_fuel2, x, y)
    df.loc[i, "fuel_a"] = coef[0]
    df.loc[i, "fuel_b"] = coef[1]

    # print(r["name"], coef)
    # xx = np.linspace(-1, 1, 100)
    # plt.plot(xx, func_fuel(xx, *coef))
    # plt.scatter(x, y)
    # plt.draw()
    # plt.waitforbuttonpress(-1)
    # plt.clf()


dfcr = pd.read_csv("input/engine_cruise_performance.csv")
df = df.merge(dfcr, how="left", left_on="name", right_on="engine")
df = df.drop("engine", axis=1)


# def func_co(x, beta, gamma):
#     return beta * (x - 0.001) ** (-gamma) * np.exp(-2 * (x - 0.001) ** beta)


# def func_nox(x, c1, p1):
#     return c1 * x ** p1


# def func_hc(x, beta, gamma):
#     return beta * (x + 0.05) ** (-gamma) * np.exp(-4 * (x - 0.001) ** beta)


# for i, r in df.iterrows():

#     # process NOx
#     x_nox = [0, r["ff_idl"], r["ff_app"], r["ff_co"], r["ff_to"]]
#     y_nox = [0, r["ei_nox_idl"], r["ei_nox_app"], r["ei_nox_co"], r["ei_nox_to"]]

#     guess = np.array([20, 0.75])
#     coef_nox, cov = curve_fit(func_nox, x_nox, y_nox, guess)

#     df.loc[i, "nox_c"] = coef_nox[0]
#     df.loc[i, "nox_p"] = coef_nox[1]

#     # process CO
#     x_co = [r["ff_idl"], r["ff_app"], r["ff_co"], r["ff_to"]]
#     y_co = [r["ei_co_idl"], r["ei_co_app"], r["ei_co_co"], r["ei_co_to"]]
#     y_co = np.maximum(y_co, [1e-7, 1e-7, 1e-7, 1e-7])

#     coef_co, covco = curve_fit(func_co, x_co[0:2], y_co[0:2])
#     df.loc[i, "co_beta"] = coef_co[0]
#     df.loc[i, "co_gamma"] = coef_co[1]
#     df.loc[i, "co_max"] = 2 * y_co[0]
#     df.loc[i, "co_min"] = (y_co[2] + y_co[3]) / 2

#     # process HC
#     x_hc = [r["ff_idl"], r["ff_app"], r["ff_co"], r["ff_to"]]
#     y_hc = [r["ei_hc_idl"], r["ei_hc_app"], r["ei_hc_co"], r["ei_hc_to"]]

#     if max(y_hc) == 0:
#         df.loc[i, "hc_na"] = True
#     else:
#         df.loc[i, "hc_na"] = False

#         df.loc[i, "hc_max"] = 2 * max(y_hc[0], y_hc[1])
#         df.loc[i, "hc_min"] = (y_hc[2] + y_hc[3]) / 2

#         y_hc = np.maximum(y_hc, [1e-7, 1e-7, 1e-7, 1e-7])

#         logX, logY = np.log10(x_hc), np.log10(y_hc)  # current powers
#         b2 = (logY[3] + logY[2]) / 2  # Average power
#         a1 = (logY[1] - logY[0]) / (logX[1] - logX[0])
#         b1 = logY[0] - a1 * logX[0]

#         x_intersect = 10 ** ((b2 - b1) / a1)
#         if x_intersect > x_hc[2]:
#             df.loc[i, "hc_ff85"] = x_hc[2]
#         else:
#             df.loc[i, "hc_ff85"] = 0

#         try:
#             coef_hc, covhc = curve_fit(func_hc, x_hc, y_hc)
#             df.loc[i, "hc_beta"] = coef_hc[0]
#             df.loc[i, "hc_gamma"] = coef_hc[1]
#             df.loc[i, "hc_a1"] = None
#             df.loc[i, "hc_b1"] = None
#             df.loc[i, "hc_b2"] = None
#         except RuntimeError:
#             df.loc[i, "hc_beta"] = None
#             df.loc[i, "hc_gamma"] = None
#             df.loc[i, "hc_a1"] = a1
#             df.loc[i, "hc_b1"] = b1
#             df.loc[i, "hc_b2"] = b2


df.to_csv("db/engines.csv", float_format="%g", index=False)
