"""OpenAP FuelFlow model."""

import importlib
import os
from typing import Any, Callable, Optional

import pandas as pd
from openap import prop
from openap.backends import BackendType

# Type alias for numeric inputs (scalar, array, or symbolic)
Numeric = Any
from openap.extra import ndarrayconvert
from openap.extra.aero import fpm, kts

from .base import FuelFlowBase


class FuelFlow(FuelFlowBase):
    """Fuel flow model based on ICAO emission databank."""

    def __init__(
        self,
        ac: str,
        eng: Optional[str] = None,
        backend: Optional[BackendType] = None,
        **kwargs,
    ):
        """Initialize FuelFlow object.

        Args:
            ac: ICAO aircraft type (for example: A320).
            eng: Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                by in the aircraft database.
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        super().__init__(ac, eng, backend=backend, **kwargs)

        if not hasattr(self, "Thrust"):
            self.Thrust = importlib.import_module("openap.thrust").Thrust

        if not hasattr(self, "Drag"):
            self.Drag = importlib.import_module("openap.drag").Drag

        if not hasattr(self, "WRAP"):
            self.WRAP = importlib.import_module("openap.kinematic").WRAP

        self.use_synonym = kwargs.get("use_synonym", False)

        self.aircraft = prop.aircraft(self.ac, **kwargs)

        if eng is None:
            eng = self.aircraft["engine"]["default"]

        self.engine_type = eng.upper()

        self.engine = prop.engine(eng)

        # Pass backend to Thrust and Drag
        self.thrust = self.Thrust(ac, eng, backend=self.backend, **kwargs)
        self.drag = self.Drag(ac, backend=self.backend, **kwargs)
        self.wrap = self.WRAP(ac, **kwargs)

        self.func_fuel = self._load_fuel_model()

    def _load_fuel_model(self) -> Callable[[Numeric], Numeric]:
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

        b = self.backend
        return lambda x: scale * (
            c1 - b.exp(-c2 * (x * b.exp(c3 * x) - b.log(c1) / c2))
        )

    @ndarrayconvert
    def at_thrust(self, total_ac_thrust: Numeric) -> Numeric:
        """Compute the fuel flow at a given total thrust.

        Args:
            total_ac_thrust: The total net thrust of the aircraft (unit: N).

        Returns:
            Fuel flow (unit: kg/s).

        """
        b = self.backend

        max_eng_thrust = self.engine["max_thrust"]
        n_eng = self.aircraft["engine"]["number"]

        ratio = (total_ac_thrust / n_eng) / max_eng_thrust

        # the approximation formula for limits:
        # x = x_min+(log(1+exp(k(x-x_min)))-log(1+exp(k(x-x_max))))/log(1+exp(k))
        # where x_min, x_max - lower and upper bounds
        # and k - sharpness of the corners

        # limit the lowest/highest ratio to 0.03/1 without creating discontinuity
        ratio = (
            (
                b.log(1 + b.exp(50 * (ratio - 0.03)))
                - b.log(1 + b.exp(45 * (ratio - 1.2)))
            )
            / (b.log(1 + b.exp(50)))
        ) + 0.03

        fuelflow = self.func_fuel(ratio) * n_eng

        return fuelflow

    @ndarrayconvert
    def takeoff(
        self,
        tas: Numeric,
        alt: Optional[Numeric] = None,
        throttle: Numeric = 1,
    ) -> Numeric:
        """Compute the fuel flow at takeoff.

        The net thrust is first estimated based on the maximum thrust model
        and throttle setting. Then FuelFlow.at_thrust() is called to compted
        the thrust.

        Args:
            tas: Aircraft true airspeed (unit: kt).
            alt: Altitude of airport (unit: ft). Defaults to sea-level.
            throttle: The throttle setting, between 0 and 1.
                Defaults to 1, which is at full thrust.

        Returns:
            Fuel flow (unit: kg/s).

        """
        Tmax = self.thrust.takeoff(tas=tas, alt=alt)
        fuelflow = throttle * self.at_thrust(Tmax)
        return fuelflow

    @ndarrayconvert
    def enroute(
        self,
        mass: Numeric,
        tas: Numeric,
        alt: Numeric,
        vs: Numeric = 0,
        acc: Numeric = 0,
        dT: Numeric = 0,
        limit: bool = True,
    ) -> Numeric:
        """Compute the fuel flow during climb, cruise, or descent.

        The net thrust is first estimated based on the dynamic equation.
        Then FuelFlow.at_thrust() is called to compted the thrust. Assuming
        no flap deflection and no landing gear extended.

        Args:
            mass: Aircraft mass (unit: kg).
            tas: Aircraft true airspeed (unit: kt).
            alt: Aircraft altitude (unit: ft).
            vs: Vertical rate (unit: ft/min). Default is 0.
            acc: Acceleration (unit: m/s^2). Default is 0.
            dT: Temperature shift (unit: K or degC). Default is 0.
            limit: Apply thrust limits. Default is True.

        Returns:
            Fuel flow (unit: kg/s).

        """
        b = self.backend

        D = self.drag.clean(mass=mass, tas=tas, alt=alt, vs=vs, dT=dT)

        gamma = b.arctan2(vs * fpm, tas * kts)

        T = D + mass * 9.81 * b.sin(gamma) + mass * acc

        fuelflow = self.at_thrust(T)

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

        xx = self.backend.linspace(0, 1, 50)
        yy = self.func_fuel(xx)
        plt.plot(xx, yy, "--", color="gray")

        if plot:
            plt.show()
        else:
            return plt
