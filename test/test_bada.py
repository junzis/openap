# %%
import matplotlib.pyplot as plt

import numpy as np
import openap
import pandas as pd
from openap.addon import bada3, bada4

bada3_path = "../../../../data/bada_312"
bada4_path = "../../../../data/bada_4.2/tables"

# %%
drag = bada3.Drag("A320", bada3_path)
print("bada 3 drag", drag.clean(60000, 300, 12_000))

drag = bada4.Drag("A320-231", bada4_path)
print("bada 4 drag", drag.clean(60000, 300, 12_000))

drag = openap.Drag("A320")
print("openap drag", drag.clean(60000, 300, 12_000))

# %%

fuel_bada3 = bada3.FuelFlow("A320", bada3_path)
print("bada 3 fuel", fuel_bada3.enroute(mass=60000, tas=350, alt=35_000))

fuel_bada4 = bada4.FuelFlow("A320-231", bada4_path)
print("bada 4 fuel", fuel_bada4.enroute(mass=60000, tas=350, alt=35_000))

fuel_openap = openap.FuelFlow("A320")
print("openap fuel", fuel_openap.enroute(mass=60000, tas=350, alt=35_000))

# %%


fuel_bada3 = bada3.FuelFlow("A320", bada3_path)
fuel_bada4 = bada4.FuelFlow("A320-231", bada4_path)
fuel_openap = openap.FuelFlow("A320")

drag_bada3 = bada3.Drag("A320", bada3_path)
drag_bada4 = bada4.Drag("A320-231", bada4_path)
drag_openap = openap.Drag("A320")


flight = pd.read_csv("../examples/data/flight_a320_qar.csv").query("ALTI_STD_FT>100")


drag_estimate_bada3 = drag_bada3.clean(
    flight["MASS_KG"],
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
)

drag_estimate_bada4 = drag_bada4.clean(
    flight["MASS_KG"],
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
)

drag_estimate_openap = drag_openap.clean(
    flight["MASS_KG"],
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
)

fuel_estimate_bada3 = fuel_bada3.enroute(
    flight["MASS_KG"],
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
    flight["VERT_SPD_FTMN"],
)

fuel_estimate_bada4 = fuel_bada4.enroute(
    flight["MASS_KG"],
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
    flight["VERT_SPD_FTMN"],
)

fuel_estimate_openap = fuel_openap.enroute(
    flight["MASS_KG"],
    flight["TRUE_AIR_SPD_KT"],
    flight["ALTI_STD_FT"],
    flight["VERT_SPD_FTMN"],
)


# %%

plt.plot(flight["FLIGHT_TIME"], drag_estimate_openap, label="OpenAP drag")
plt.plot(flight["FLIGHT_TIME"], drag_estimate_bada3, label="BADA3 drag")
plt.plot(flight["FLIGHT_TIME"], drag_estimate_bada4, label="BADA4 drag")
plt.legend()
plt.ylim(0)
plt.show()


plt.plot(flight["FLIGHT_TIME"], fuel_estimate_openap * 3600, label="OpenAP fuel", lw=1)
plt.plot(flight["FLIGHT_TIME"], fuel_estimate_bada3 * 3600, label="BADA3 fuel", lw=1)
plt.plot(flight["FLIGHT_TIME"], fuel_estimate_bada4 * 3600, label="BADA4 fuel", lw=1)
plt.plot(flight["FLIGHT_TIME"], flight["FUEL_FLOW_KGH"] * 2, label="QAR fuel", lw=1)
plt.ylim(0)
# plt.ylim(2000, 4000)

plt.legend()
plt.show()

# %%

# %%
