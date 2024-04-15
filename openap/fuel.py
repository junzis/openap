"""OpenAP FuelFlow model."""

import importlib

from openap import prop
from openap.extra import ndarrayconvert


def func_fuel2(a, b):
    return lambda x: a * (x + b) ** 2


def func_fuel3(c3, c2, c1):
    return lambda x: c3 * x**3 + c2 * x**2 + c1 * x


class FuelFlow(object):
    """Fuel flow model based on ICAO emission databank."""

    def __init__(self, ac, eng=None, **kwargs):
        """Initialize FuelFlow object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            eng (string): Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                by in the aircraft database.
            polydeg (int): Order of the polynomials for fuel flow model
                            (2 or 3), defaults to 2.

        """
        if not hasattr(self, "np"):
            self.np = importlib.import_module("numpy")

        if not hasattr(self, "Thrust"):
            self.Thrust = importlib.import_module("openap.thrust").Thrust

        if not hasattr(self, "Drag"):
            self.Drag = importlib.import_module("openap.drag").Drag

        if not hasattr(self, "WRAP"):
            self.WRAP = importlib.import_module("openap.kinematic").WRAP

        self.aircraft = prop.aircraft(ac, **kwargs)

        if eng is None:
            eng = self.aircraft["engine"]["default"]

        self.engine = prop.engine(eng)

        self.thrust = self.Thrust(ac, eng, **kwargs)
        self.drag = self.Drag(ac, **kwargs)
        self.wrap = self.WRAP(ac, **kwargs)

        polydeg = kwargs.get("polydeg", 2)

        if polydeg == 2:
            a, b = self.engine["fuel_a"], self.engine["fuel_b"]
            self.polyfuel = func_fuel2(a, b)
        elif polydeg == 3:
            c3, c2, c1 = (
                self.engine["fuel_c3"],
                self.engine["fuel_c2"],
                self.engine["fuel_c1"],
            )
            self.polyfuel = func_fuel3(c3, c2, c1)
        else:
            raise ValueError(f"Polydeg must be 2 or 3")

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
        n_eng = self.aircraft["engine"]["number"]
        engthr = acthr / n_eng

        maxthr = self.thrust.takeoff(tas=0, alt=0)
        ratio = acthr / maxthr

        if limit:
            # assume minimum is 7% at idle
            ratio = self.np.where(ratio < 0.07, 0.07, ratio)
            ratio = self.np.where(ratio > 1, 1, ratio)

        ff_sl = self.polyfuel(ratio)
        ff_corr_alt = self.engine["fuel_ch"] * (engthr / 1000) * (alt * 0.3048)
        ff_eng = ff_sl + ff_corr_alt

        fuelflow = ff_eng * n_eng

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
    def enroute(self, mass, tas, alt, path_angle=0, limit=True):
        """Compute the fuel flow during climb, cruise, or descent.

        The net thrust is first estimated based on the dynamic equation.
        Then FuelFlow.at_thrust() is called to compted the thrust. Assuming
        no flap deflection and no landing gear extended.

        Args:
            mass (int or ndarray): Aircraft mass (unit: kg).
            tas (int or ndarray): Aircraft true airspeed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).
            path_angle (float or ndarray): Flight path angle (unit: degrees).

        Returns:
            float: Fuel flow (unit: kg/s).

        """
        D = self.drag.clean(mass=mass, tas=tas, alt=alt, path_angle=path_angle)

        gamma = path_angle * 3.142 / 180  # radians

        T = D + mass * 9.81 * self.np.sin(gamma)

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
