# %%
import pathlib
import warnings
from glob import glob

import matplotlib.pyplot as plt
from acropole import FuelEstimator
from scipy.optimize import curve_fit
from sklearn import cluster
from tqdm.autonotebook import tqdm
from traffic.core import Flight, Traffic

import numpy as np
import openap
import pandas as pd

pd.options.display.max_columns = 100

warnings.filterwarnings("ignore")

# %%

wave_drag = True
show_plot = True


data_dir = "/mnt/md16/data/openap-rebuild/2022/"

# %%
# def func_old(ratio, coef):
#     return -coef * (ratio - 1) ** 2 + coef

# def func_try_1(x, a, b, c):
#     return a / (b + np.exp(-c * x) / x)


# def func(x, c1, c2, c3):
#     return c1 / (1 + np.exp(-c2 * x + c3))


def func(x, c1, c2, c3):
    return c1 - np.exp(-c2 * (x * np.exp(c3 * x) - np.log(c1) / c2))


fe = FuelEstimator()

acropole_aircraft = pd.read_csv("acropole_aircraft_params.csv")

acropole_typecodes = acropole_aircraft.ACFT_ICAO_TYPE.unique()

# acropole_typecodes = ["CRJ9", "A318"]

rng = np.random.default_rng(42)

results = []

for typecode in acropole_typecodes:
    if typecode == "B734":  # model is bad
        continue

    if typecode.lower() not in openap.prop.available_aircraft():
        print(f"{typecode} not in openap")
        continue

    aac = acropole_aircraft.query(f"ACFT_ICAO_TYPE=='{typecode.upper()}'")
    if aac.shape[0] == 0:
        print(f"{typecode} not in acropole")
        continue

    engine_type = aac.ENGINE_ICAO.iloc[0]

    ac = openap.prop.aircraft(typecode)
    eng = openap.prop.engine(ac["engine"]["default"])

    drag = openap.Drag(typecode, wave_drag=wave_drag, use_synonym=True)
    thrust = openap.Thrust(typecode, use_synonym=True)

    print(typecode, engine_type)

    typecode_file = f"{data_dir}/typecodes/{typecode.lower()}.parquet"

    if pathlib.Path(typecode_file).exists():
        df_typecode = pd.read_parquet(typecode_file)

    else:
        files_all_flights = rng.permutation(glob(f"{data_dir}/full_flights/*.parquet"))

        t_typecode = None

        for i, f in tqdm(enumerate(files_all_flights)):
            t = Traffic.from_file(f).assign(flight_id=lambda d: d.flight_id + f"_{i}")

            t_select = (
                t.query(f"typecode=='{typecode.lower()}'").longer_than("2h").eval()
            )

            if t_typecode is None:
                t_typecode = t_select
            else:
                t_typecode = t_typecode + t_select

            n_flights = len(t_typecode.flight_ids) if t_typecode is not None else 0
            print(f"got {n_flights} flights | {pathlib.Path(f).stem}")

            if t_typecode is None and i > 5:
                break

            if t_typecode is not None and len(t_typecode.flight_ids) > 500:
                t_typecode = t_typecode.sample(500)
                break

        if t_typecode is None:
            print(f"{typecode} no flight data available")
            continue

        typecode_flights = []

        for flight in tqdm(t_typecode, desc="processing flights"):
            mass = rng.uniform(ac["limits"]["MTOW"] * 0.7, ac["limits"]["MTOW"] * 0.95)

            df = flight.resample("4s").data.assign(
                typecode=typecode.upper(),
                second=lambda d: (d.timestamp - d.timestamp.iloc[0]).dt.total_seconds(),
                v=lambda d: d.groundspeed * openap.aero.kts,
                trk=lambda d: np.radians(d.track),
                vgx=lambda d: d.v * np.sin(d.trk),
                vgy=lambda d: d.v * np.cos(d.trk),
                vax=lambda d: d.vgx - d.u_component_of_wind,
                vay=lambda d: d.vgy - d.v_component_of_wind,
                tas=lambda d: np.sqrt(d.vax**2 + d.vay**2) / openap.aero.kts,
            )

            df = fe.estimate(
                df,
                timestamp="second",
                airspeed="tas",
            )

            # plt.plot(flight_fuel.second, flight_fuel.fuel_flow)
            # plt.show()

            D = drag.clean(
                mass=mass,
                tas=df.tas,
                alt=df.altitude,
                vs=df.vertical_rate,
            )

            T = D + mass * 9.81 * np.sin(
                np.arctan2(
                    df.vertical_rate.values * openap.aero.fpm,
                    df.tas.values * openap.aero.kts,
                )
            )

            T0 = eng["max_thrust"] * ac["engine"]["number"]

            T_idle = thrust.descent_idle(tas=df.tas, alt=df.altitude)
            T_max = thrust.climb(alt=df.altitude, tas=df.tas, roc=df.vertical_rate)

            mask = (T > T_idle) & (T < T_max)

            df_ = df[mask]
            ratio_samples = T[mask] / T0

            df_ = df_.assign(thrust_ratio=ratio_samples).query("0<thrust_ratio<1")

            typecode_flights.append(df_)

        df_typecode = (
            pd.concat(typecode_flights, ignore_index=True)
            .assign(thrust_ratio=lambda d: d.thrust_ratio.round(2))
            .assign(fuel_flow=lambda d: d.fuel_flow.round(2))
            .drop_duplicates(subset=["thrust_ratio", "fuel_flow"])
        )

        df_typecode.to_parquet(typecode_file, index=False)

    # remove outliers

    df_typecode = df_typecode.assign(
        engine_fuel_flow=lambda d: d.fuel_flow.round(2) / ac["engine"]["number"]
    ).sample(10000, replace=True)

    X = np.array([df_typecode.thrust_ratio, df_typecode.engine_fuel_flow]).T
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    clustering = cluster.DBSCAN(eps=0.1, min_samples=100).fit(X)
    mask = clustering.labels_ == 0

    df0 = (
        df_typecode[mask]
        .assign(alt_=lambda d: d.altitude // 2000 * 2000)
        .assign(tas_=lambda d: d.tas // 20 * 20)
    )

    # fmt: off
    df1 = df0.query("vertical_rate<=-1000")
    df2 = df0.query("-250<=vertical_rate<=250")
    df3 = df0.query("vertical_rate>=500 and altitude>15000")
    df4 = df0.query("vertical_rate>=1000 and altitude<10000 and tas<180")
    df5 = df0.query("vertical_rate>=1000 and altitude<10000 and tas>250")
    # fmt: on

    df_combine = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

    popt, pcov = curve_fit(
        func,
        df_combine.thrust_ratio,
        df_combine.engine_fuel_flow,
        bounds=[(0, 0, 0), (df_combine.engine_fuel_flow.max() * 1.2, np.inf, np.inf)],
    )

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        ratio = np.linspace(0, 1, 100)
        total_ac_fuel = func(ratio, *popt) * ac["engine"]["number"]

        ax1.scatter(df0.thrust_ratio, df0.fuel_flow, alpha=0.1, s=5)
        ax1.plot(ratio, total_ac_fuel, "r-", label=f"fit: {popt}")
        ax1.legend()
        ax1.set_xlim(0, 0.7)
        ax1.set_xlabel("Thrust ratio")
        ax1.set_ylabel("Fuel flow")

        ax2.scatter(df1.thrust_ratio, df1.fuel_flow, s=5, alpha=0.1, c="tab:green")
        ax2.scatter(df2.thrust_ratio, df2.fuel_flow, s=5, alpha=0.1, c="tab:orange")
        ax2.scatter(df3.thrust_ratio, df3.fuel_flow, s=5, alpha=0.1, c="tab:purple")
        ax2.scatter(df4.thrust_ratio, df4.fuel_flow, s=5, alpha=0.1, c="tab:red")
        ax2.scatter(df5.thrust_ratio, df5.fuel_flow, s=5, alpha=0.1, c="gray")
        ax2.plot(ratio, total_ac_fuel, "r-")
        ax2.set_xlim(0, 0.7)
        ax2.set_xlabel("Thrust ratio")
        ax2.set_ylabel("Fuel flow")

        plt.suptitle(typecode)
        plt.tight_layout()
        plt.show()

    results.append(
        {
            "typecode": typecode,
            "engine_type": engine_type,
            "c1": popt[0],
            "c2": popt[1],
            "c3": popt[2],
        }
    )


# %%
# import seaborn as sns

# sns.relplot(
#     data=df3,
#     x="thrust_ratio",
#     y="fuel_flow",
#     # col="alt_",
#     # row="tas_",
#     col="tas_",
#     col_wrap=3,
#     kind="scatter",
#     alpha=0.02,
#     edgecolor=None,
# )

# plt.xlim(0, 1)
# plt.ylim(0, 2.2)

# %%
fuel_models = pd.DataFrame(results)


# build default model when aircraft is not available
# fuel flow normalized by takeoff fuel flow, ff_to

ff_norms = []

ratio_samples = np.arange(0, 1, 0.01)

for i, row in fuel_models.iterrows():
    coef = row[["c1", "c2", "c3"]].to_list()

    fuel = func(ratio_samples, *coef)

    typecode = row["typecode"]
    ff_to = openap.prop.engine(row["engine_type"])["ff_to"]

    ff_norm = fuel / ff_to

    ff_norms.append(ff_norm)

    plt.plot(ratio_samples, ff_norm, label=row["typecode"], alpha=0.5)

df = pd.DataFrame(
    np.array(ff_norms), index=fuel_models.typecode.to_list(), columns=ratio_samples
)

plt.scatter(ratio_samples, df.mean(axis=0), label="mean", color="b", s=5)

popt, pcov = curve_fit(func, ratio_samples, df.mean(axis=0))

print("default", popt)

x = np.linspace(0, 1, 100)
plt.plot(x, func(x, *popt), "b-")
plt.xlabel("Thrust ratio")
plt.ylabel("Fuel flow")


plt.legend(ncol=3)
plt.show()


# %%

fuel_models = pd.concat(
    [
        fuel_models,
        pd.DataFrame(
            [
                {
                    "typecode": "default",
                    "engine_type": "default",
                    "c1": popt[0],
                    "c2": popt[1],
                    "c3": popt[2],
                }
            ]
        ),
    ],
    ignore_index=True,
)

# %%

fuel_models.to_csv("fuel_models.csv", index=False)

# %%
