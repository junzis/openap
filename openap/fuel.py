"""OpenAP FuelFlow model."""

import glob
import importlib
import os
import pathlib

import pandas as pd
import yaml
from openap import aero, prop
from openap.extra import ndarrayconvert

curr_path = os.path.dirname(os.path.realpath(__file__))
dir_fuelmodel = os.path.join(curr_path, "data/fuel/")
file_synonym = os.path.join(curr_path, "data/fuel/_synonym.csv")

fuel_synonym = pd.read_csv(file_synonym)


def func_fuel(coef):
    return lambda x: -coef * (x - 1) ** 2 + coef


class FuelFlow(object):
    """Fuel flow model based on ICAO emission databank."""

    def __init__(self, ac, eng=None, **kwargs):
        """Initialize FuelFlow object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            eng (string): Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                by in the aircraft database.

        """
        if not hasattr(self, "np"):
            self.np = importlib.import_module("numpy")

        if not hasattr(self, "Thrust"):
            self.Thrust = importlib.import_module("openap.thrust").Thrust

        if not hasattr(self, "Drag"):
            self.Drag = importlib.import_module("openap.drag").Drag

        if not hasattr(self, "WRAP"):
            self.WRAP = importlib.import_module("openap.kinematic").WRAP

        self.use_synonym = kwargs.get("use_synonym", False)

        self.ac = ac.lower()
        self.aircraft = prop.aircraft(ac, **kwargs)

        if eng is None:
            eng = self.aircraft["engine"]["default"]

        self.engine = prop.engine(eng)

        self.thrust = self.Thrust(ac, eng, **kwargs)
        self.drag = self.Drag(ac, **kwargs)
        self.wrap = self.WRAP(ac, **kwargs)

        self.params = self.fuel_params()
        self.polyfuel = func_fuel(self.params["fuel_coef"])

    def fuel_params(self):
        """Obtain the fuel model parameters.

        Returns:
            dict: fuel model parameters.
        """
        polar_files = glob.glob(dir_fuelmodel + "*.yml")
        ac_polar_available = [pathlib.Path(s).stem for s in polar_files]

        if self.ac in ac_polar_available:
            ac = self.ac
        else:
            syno = fuel_synonym.query("orig==@self.ac")
            if self.use_synonym and syno.shape[0] > 0:
                ac = syno.new.iloc[0]
            else:
                raise ValueError(f"Fuel parameters for {self.ac} not available.")

        f = dir_fuelmodel + ac + ".yml"
        with open(f, "r") as file:
            params = yaml.safe_load(file.read())
        return params

    @ndarrayconvert
    def at_thrust(self, acthr, alt=0, limit=True):
        """Compute the fuel flow at a given total thrust.

        Args:
            acthr (int or ndarray): The total net thrust of the
                aircraft (unit: N).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: Fuel flow (unit: kg/s).

        """
        max_eng_thrust = self.engine["max_thrust"]
        n_eng = self.aircraft["engine"]["number"]

        ratio = acthr / (max_eng_thrust * n_eng)

        if limit:
            ratio = self.np.where(ratio < 0, 0, ratio)
            ratio = self.np.where(ratio > 1, 1, ratio)

        fuelflow = self.polyfuel(ratio)

        return fuelflow

    @ndarrayconvert
    def takeoff(self, tas, alt=None, throttle=1):
        """Compute the fuel flow at takeoff.

        The net thrust is first estimated based on the maximum thrust model
        and throttle setting. Then FuelFlow.at_thrust() is called to compted
        the thrust.

        Args:
            tas (int or ndarray): Aircraft true airspeed (unit: kt).
            alt (int or ndarray): Altitude of airport (unit: ft).
                Defaults to sea-level.
            throttle (float or ndarray): The throttle setting, between 0 and 1.
                Defaults to 1, which is at full thrust.

        Returns:
            float: Fuel flow (unit: kg/s).

        """
        Tmax = self.thrust.takeoff(tas=tas, alt=alt)
        fuelflow = throttle * self.at_thrust(Tmax)
        return fuelflow

    @ndarrayconvert
    def enroute(self, mass, tas, alt, vs=0, acc=0, limit=True):
        """Compute the fuel flow during climb, cruise, or descent.

        The net thrust is first estimated based on the dynamic equation.
        Then FuelFlow.at_thrust() is called to compted the thrust. Assuming
        no flap deflection and no landing gear extended.

        Args:
            mass (int or ndarray): Aircraft mass (unit: kg).
            tas (int or ndarray): Aircraft true airspeed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).
            vs (float or ndarray): Vertical rate (unit: ft/min). Default is 0.
            acc (float or ndarray): acceleration (unit: m/s^2). Default is 0.

        Returns:
            float: Fuel flow (unit: kg/s).

        """
        D = self.drag.clean(mass=mass, tas=tas, alt=alt, vs=vs)

        gamma = self.np.arctan2(vs * aero.fpm, tas * aero.kts)

        if limit:
            # limit gamma to -10 to 10 degrees (0.175 radians)
            gamma = self.np.where(gamma < -0.175, -0.175, gamma)
            gamma = self.np.where(gamma > 0.175, 0.175, gamma)

            # limit acc to 5 m/s^2
            acc = self.np.where(acc < -5, -5, acc)
            acc = self.np.where(acc > 5, 5, acc)

        T = D + mass * 9.81 * self.np.sin(gamma) + mass * acc

        if limit:
            T_max = self.thrust.climb(tas=tas, alt=alt, roc=0)
            T_idle = self.thrust.descent_idle(tas=tas, alt=alt)

            # below idle thrust
            T = self.np.where(T < T_idle, T_idle, T)

            # outside performance boundary (with margin of 20%)
            T = self.np.where(T > 1.2 * T_max, 1.2 * T_max, T)

        fuelflow = self.at_thrust(T, alt, limit=limit)

        return fuelflow

    def plot_model(self, plot=True):
        """Plot the engine fuel model, or return the pyplot object.

        Args:
            plot (bool): Display the plot or return an object.

        Returns:
            None or pyplot object.

        """
        import matplotlib.pyplot as plt

        x = [0.07, 0.3, 0.85, 1.0]
        y = [
            self.engine["ff_idl"],
            self.engine["ff_app"],
            self.engine["ff_co"],
            self.engine["ff_to"],
        ]
        plt.scatter(x, y, color="k")

        xx = self.np.linspace(0, 1, 50)
        yy = self.polyfuel(xx)
        plt.plot(xx, yy, "--", color="gray")

        if plot:
            plt.show()
        else:
            return plt
