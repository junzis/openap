""" "OpenAP FuelFlow model."""

import glob
import importlib
import os
import pathlib

import yaml

import pandas as pd
from openap import prop
from openap.extra import ndarrayconvert
from openap.extra.aero import fpm, kts

from .base import FuelFlowBase


class FuelFlow(FuelFlowBase):
    """Fuel flow model based on ICAO emission databank."""

    def __init__(self, ac, eng=None, **kwargs):
        """Initialize FuelFlow object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            eng (string): Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                by in the aircraft database.

        """
        super().__init__(ac, eng, **kwargs)

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

        self.engine_type = eng.upper()

        self.engine = prop.engine(eng)

        self.thrust = self.Thrust(ac, eng, **kwargs)
        self.drag = self.Drag(ac, **kwargs)
        self.wrap = self.WRAP(ac, **kwargs)

        self.func_fuel = self._load_fuel_model()

    def _load_fuel_model(self) -> dict:
        curr_path = os.path.dirname(os.path.realpath(__file__))
        file_fuel_models = os.path.join(curr_path, "data/fuel/fuel_models.csv")

        fuel_models = pd.read_csv(file_fuel_models).assign(
            typecode=lambda d: d.typecode.str.lower()
        )

        if self.ac in fuel_models.typecode.values:
            ac = self.ac
        else:
            ac = "default"

        params = fuel_models.query(f"typecode=='{ac}'").iloc[0].to_dict()

        c1, c2, c3 = params["c1"], params["c2"], params["c3"]

        scale = 1

        if ac == "default":
            scale = self.engine["ff_to"]
        elif self.engine_type != params["engine_type"].upper():
            ref_engine = prop.engine(params["engine_type"])
            scale = self.engine["ff_to"] / ref_engine["ff_to"]

        return (
            lambda x: c1
            - self.sci.exp(-c2 * (x * self.sci.exp(c3 * x) - self.sci.log(c1) / c2))
            * scale
        )

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

        # the approximation formula for limits:
            # x = x_min+(log(1+exp(k(x-x_min)))-log(1+exp(k(x-x_max))))/log(1+exp(k))
            # where x_min, x_max - lower and upper bounds
            # and k - sharpness of the corners

        # always limit the lowest and highest ratio to 0.03 and 1 without creating a discontinuity
        ratio = (
            (
                self.sci.log(1 + self.sci.exp(50 * (ratio - 0.03)))
                - self.sci.log(1 + self.sci.exp(50 * (ratio - 1)))
            )
            / (self.sci.log(1 + self.sci.exp(50)))
        ) + 0.03

        fuelflow = self.func_fuel(ratio)

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
    def enroute(self, mass, tas, alt, vs=0, acc=0, dT=0, limit=True):
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
            dT (float or ndarray): Temperature shift (unit: K or degC),default = 0
        Returns:
            float: Fuel flow (unit: kg/s).

        """
        D = self.drag.clean(mass=mass, tas=tas, alt=alt, vs=vs, dT=dT)

        gamma = self.sci.arctan2(vs * fpm, tas * kts)

        if limit:
            # limit gamma to -20 to 20 degrees (0.175 radians)
            # gamma = self.sci.where(gamma < -0.175, -0.175, gamma)
            # gamma = self.sci.where(gamma > 0.175, 0.175, gamma)
            gamma = (
                (
                    self.sci.log(1 + self.sci.exp(100 * (gamma + 0.175)))
                    - self.sci.log(1 + self.sci.exp(100 * (gamma - 0.175)))
                )
                / (self.sci.log(1 + self.sci.exp(100)))
            ) - 0.175

            # limit acc to 5 m/s^2
            # acc = self.sci.where(acc < -5, -5, acc)
            # acc = self.sci.where(acc > 5, 5, acc)
            acc = (
                (
                    self.sci.log(1 + self.sci.exp(50 * (acc + 5)))
                    - self.sci.log(1 + self.sci.exp(50 * (acc - 5)))
                )
                / (self.sci.log(1 + self.sci.exp(50)))
            ) - 5
            
        T = D + mass * 9.81 * self.sci.sin(gamma) + mass * acc

        if limit:
            T_max = self.thrust.climb(tas=tas, alt=alt, roc=0, dT=dT)
            T_idle = self.thrust.descent_idle(tas=tas, alt=alt, dT=dT)

            # # below idle thrust (with margin of 20%)
            # T = self.sci.where(T < T_idle * 0.8, T_idle * 0.8, T)

            # # outside performance boundary (with margin of 20%)
            # T = self.sci.where(T > T_max * 1.2, T_max * 1.2, T)

            T = (
                (
                    self.sci.log(1 + self.sci.exp(50 * (T - T_idle * 0.8) / 100_000))
                    - self.sci.log(1 + self.sci.exp(50 * (T - T_max * 1.2) / 100_000))
                )
                / (self.sci.log(1 + self.sci.exp(50)))
            ) * 100_000 + T_idle * 0.8

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

        xx = self.sci.linspace(0, 1, 50)
        yy = self.func_fuel(xx)
        plt.plot(xx, yy, "--", color="gray")

        if plot:
            plt.show()
        else:
            return plt
