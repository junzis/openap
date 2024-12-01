from glob import glob

import matplotlib.pyplot as plt
import yaml

import numpy as np

files = sorted(glob("../openap/data/fuel/*.yml"))


def func_fuel(x, coef):
    return -coef * (x - 1) ** 2 + coef


for file in files:
    with open(file, "r") as f:
        params = yaml.safe_load(f.read())
        name = f"{params['aircraft']} ({params['engine']})"
        # print(name)

        x = np.linspace(0, 1, 100)

        y = func_fuel(x, params["fuel_coef"])

        plt.plot(x, y, label=name)
        plt.xlabel("$T / T_{max}$")
        plt.ylabel("Fuel flow (kg/s)")

plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), ncol=1)

plt.show()
