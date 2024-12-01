# %%
import matplotlib.pyplot as plt

from openap import FlightGenerator, aero

# %%
flightgen = FlightGenerator(ac="a320")
# flightgen.enable_noise()

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
plt.suptitle("Climb trajectories")
for i in range(5):
    data = flightgen.climb(dt=10, random=True)
    ax[0][0].plot(
        data["t"],
        data["h"] / aero.ft,
        label="%d/%.2f" % (data["cas_const_cl"].iloc[0], data["mach_const_cl"].iloc[0]),
    )
    ax[0][0].set_ylabel("Altitude (ft)")
    ax[0][1].plot(data["t"], data["s"] / 1000)
    ax[0][1].set_ylabel("Distanse (km)")
    ax[1][0].plot(data["t"], data["v"] / aero.kts)
    ax[1][0].set_ylabel("True airspeed (kt)")
    ax[1][1].plot(data["t"], data["vs"] / aero.fpm)
    ax[1][1].set_ylabel("Vertical rate (ft/min)")
    ax[0][0].legend()
plt.show()

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
plt.suptitle("Descent trajectories")
for i in range(5):
    data = flightgen.descent(dt=10, random=True)
    ax[0][0].plot(
        data["t"],
        data["h"] / aero.ft,
        label="%d/%.2f" % (data["cas_const_de"].iloc[0], data["mach_const_de"].iloc[0]),
    )
    ax[0][0].set_ylabel("Altitude (ft)")
    ax[0][1].plot(data["t"], data["s"] / 1000)
    ax[0][1].set_ylabel("Distanse (km)")
    ax[1][0].plot(data["t"], data["v"] / aero.kts)
    ax[1][0].set_ylabel("True airspeed (kt)")
    ax[1][1].plot(data["t"], data["vs"] / aero.fpm)
    ax[1][1].set_ylabel("Vertical rate (ft/min)")
    ax[0][0].legend()
plt.show()

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
plt.suptitle("Cruise trajectories")
for i in range(5):
    data = flightgen.cruise(dt=60, random=True)
    ax[0][0].plot(data["t"], data["h"] / aero.ft, label="%d" % data["alt_cr"].iloc[0])
    ax[0][0].set_ylabel("Altitude (ft)")
    ax[0][1].plot(data["t"], data["s"] / 1000)
    ax[0][1].set_ylabel("Distanse (km)")
    ax[1][0].plot(data["t"], data["v"] / aero.kts)
    ax[1][0].set_ylabel("True airspeed (kt)")
    ax[1][1].plot(data["t"], data["vs"] / aero.fpm)
    ax[1][1].set_ylabel("Vertical rate (ft/min)")
    ax[0][0].legend()
plt.show()

# %%

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
plt.suptitle("Complete trajectories")
for i in range(5):
    data = flightgen.complete(dt=10, random=True)
    ax[0][0].plot(data["t"], data["h"] / aero.ft)
    ax[0][0].set_ylabel("Altitude (ft)")
    ax[0][1].plot(data["t"], data["s"] / 1000)
    ax[0][1].set_ylabel("Distanse (km)")
    ax[1][0].plot(data["t"], data["v"] / aero.kts)
    ax[1][0].set_ylabel("True airspeed (kt)")
    ax[1][1].plot(data["t"], data["vs"] / aero.fpm)
    ax[1][1].set_ylabel("Vertical rate (ft/min)")
plt.show()

# %%
