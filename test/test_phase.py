import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openap import FlightPhase

root = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(root + "/data/flight_phase_test.csv")

ts = df["ts"].values
ts = ts - ts[0]
alt = df["alt"].values
spd = df["spd"].values
roc = df["roc"].values

ts_ = np.arange(0, ts[-1], 1)
alt_ = np.interp(ts_, ts, alt)
spd_ = np.interp(ts_, ts, spd)
roc_ = np.interp(ts_, ts, roc)


fp = FlightPhase()
fp.set_trajectory(ts_, alt_, spd_, roc_)
labels = fp.phaselabel()


def test_segment():

    phasecolors = {
        "GND": "black",
        "CL": "green",
        "DE": "blue",
        "LVL": "cyan",
        "CR": "purple",
        "NA": "red",
    }

    colors = [phasecolors[lbl] for lbl in labels]

    plt.subplot(311)
    plt.scatter(ts_, alt_, marker=".", c=colors, lw=0)
    plt.ylabel("altitude (ft)")

    plt.subplot(312)
    plt.scatter(ts_, spd_, marker=".", c=colors, lw=0)
    plt.ylabel("speed (kt)")

    plt.subplot(313)
    plt.scatter(ts_, roc_, marker=".", c=colors, lw=0)
    plt.ylabel("roc (fpm)")

    plt.show()


def test_phase():
    idx = fp.flight_phase_indices()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    plt.plot(ts, alt, color="gray")
    y0, y1 = ax.get_ylim()
    for k, v in idx.items():
        if v is None:
            continue

        plt.plot([ts_[v], ts_[v]], [y0, y1], label=k)
        # plt.text(lx, ly, k, ha='center')
    plt.title("altitude")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test_segment()
    test_phase()
