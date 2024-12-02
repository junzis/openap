# %%
import matplotlib.pyplot as plt

import numpy as np
import openap
import pandas as pd
from openap.addon import bada4

bada_path = "../../../../data/bada_4.2/tables"

# %%
drag = bada4.Drag("A320", bada_path)
print("bada drag", drag.clean(60000, 300, 12_000))

drag = openap.Drag("A320")
print("openap drag", drag.clean(60000, 300, 12_000))

# %%

fuel_bada = bada4.FuelFlow("A320", bada_path)
print("bada fuel", fuel_bada.enroute(mass=60000, tas=350, alt=35_000))

fuel_openap = openap.FuelFlow("A320")
print("openap fuel", fuel_openap.enroute(mass=60000, tas=350, alt=35_000))

# %%

typecode = "a320"

fuel_bada = bada4.FuelFlow(typecode, bada_path)
fuel_openap = openap.FuelFlow(typecode)

drag_bada = bada4.Drag(typecode, bada_path)
drag_openap = openap.Drag(typecode)

mass_assume = openap.prop.aircraft(typecode)["mtow"] * 0.8


flight = pd.read_csv("../examples/data/flight_a320_qar.csv").query("ALTI_STD_FT>100")


drag_estimate_bada = drag_bada.clean(
    flight["MASS_KG"],
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
)

drag_estimate_openap = drag_openap.clean(
    flight["MASS_KG"],
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
)

plt.plot(flight["FLIGHT_TIME"], drag_estimate_bada, label="BADA4 drag")
plt.plot(flight["FLIGHT_TIME"], drag_estimate_openap, label="OpenAP drag")
plt.legend()
plt.ylim(0)
plt.show()

fuel_estimate_bada = fuel_bada.enroute(
    flight["MASS_KG"],
    # mass_assume,
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
    flight["VERT_SPD_FTMN"],
)


fuel_estimate_openap = fuel_openap.enroute(
    flight["MASS_KG"],
    # mass_assume,
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
    flight["VERT_SPD_FTMN"],
)


plt.plot(flight["FLIGHT_TIME"], flight["FUEL_FLOW_KGH"] * 2, label="QAR fuel")
plt.plot(flight["FLIGHT_TIME"], fuel_estimate_bada * 3600, label="BADA4 fuel")
plt.plot(flight["FLIGHT_TIME"], fuel_estimate_openap * 3600, label="OpenAP fuel")
plt.ylim(0)
# plt.ylim(2000, 4000)

plt.legend()
plt.show()
