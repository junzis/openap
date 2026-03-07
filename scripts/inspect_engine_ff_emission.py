import numpy as np
from openap import prop
from pprint import pprint as print
import matplotlib.pyplot as plt

acs = prop.available_aircraft()

for ac in acs:
    engs = prop.aircraft_engine_options(ac)
    for eng in engs:

        try:
            e = prop.engine(eng)
            print(e)
        except (KeyError, ValueError):
            print(f"{eng} from {ac} cannot be found.")
            continue

        c3, c2, c1 = (
            e["fuel_c3"],
            e["fuel_c2"],
            e["fuel_c1"],
        )

        x = np.array([0.07, 0.3, 0.85, 1.0])
        xx = np.linspace(0, 1, 100)

        plt.suptitle(f"{ac} / {eng}")

        plt.subplot(221)
        plt.scatter(x, [e["ff_idl"], e["ff_app"], e["ff_co"], e["ff_to"]])
        plt.plot(xx, prop.func_fuel(e["fuel_c3"], e["fuel_c2"], e["fuel_c1"])(xx))
        plt.ylabel("Fuel flow (kg)")
        plt.xlabel("Thrust ratio")

        plt.subplot(222)
        plt.plot(
            [e["ff_idl"], e["ff_app"], e["ff_co"], e["ff_to"]],
            [e["ei_nox_idl"], e["ei_nox_app"], e["ei_nox_co"], e["ei_nox_to"]],
            ".--",
        )
        plt.ylabel("EI NOx")
        plt.xlabel("Fuel flow (kg)")

        plt.subplot(223)
        plt.plot(
            [e["ff_idl"], e["ff_app"], e["ff_co"], e["ff_to"]],
            [e["ei_co_idl"], e["ei_co_app"], e["ei_co_co"], e["ei_co_to"]],
            ".--",
        )
        plt.ylabel("EI COx")
        plt.xlabel("Fuel flow (kg)")

        plt.subplot(224)
        plt.plot(
            [e["ff_idl"], e["ff_app"], e["ff_co"], e["ff_to"]],
            [e["ei_hc_idl"], e["ei_hc_app"], e["ei_hc_co"], e["ei_hc_to"]],
            ".--",
        )
        plt.ylabel("EI HCx")
        plt.xlabel("Fuel flow (kg)")

        plt.tight_layout()
        plt.draw()
        plt.waitforbuttonpress(-1)
        plt.clf()
