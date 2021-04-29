"""OpenAP FuelFlow model."""

import numpy as np
from openap.extra import aero
from openap import prop, Thrust, Drag
from openap.extra import ndarrayconvert


class FuelFlow(object):
    """Fuel flow model based on ICAO emission databank."""

    def __init__(self, ac, eng=None):
        """Initialize FuelFlow object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            eng (string): Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                by in the aircraft database.

        """
        self.aircraft = prop.aircraft(ac)

        if eng is None:
            eng = self.aircraft["engine"]["default"]

        self.engine = prop.engine(eng)

        self.thrust = Thrust(ac, eng)
        self.drag = Drag(ac)

        c3, c2, c1 = (
            self.engine["fuel_c3"],
            self.engine["fuel_c2"],
            self.engine["fuel_c1"],
        )
        # print(c3,c2,c1)

        self.fuel_flow_model = lambda x: c3 * x ** 3 + c2 * x ** 2 + c1 * x

    # @ndarrayconvert
    def at_thrust(self, acthr, alt=0):
        """Compute the fuel flow at a given total thrust.

        Args:
            acthr (int or ndarray): The total net thrust of the aircraft (unit: N).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: Fuel flow (unit: kg/s).

        """
        n_eng = self.aircraft["engine"]["number"]
        engthr = acthr / n_eng

        ratio = engthr / self.engine["max_thrust"]

        ff_sl = self.fuel_flow_model(ratio)
        ff_corr_alt = self.engine["fuel_ch"] * (engthr / 1000) * (alt * aero.ft)
        ff_eng = ff_sl + ff_corr_alt

        fuelflow = ff_eng * n_eng

        return fuelflow

    # @ndarrayconvert
    def takeoff(self, tas, alt=None, throttle=1):
        """Compute the fuel flow at takeoff.

        The net thrust is first estimated based on the maximum thrust model
        and throttle setting. Then FuelFlow.at_thrust() is called to compted
        the thrust.

        Args:
            tas (int or ndarray): Aircraft true airspeed (unit: kt).
            alt (int or ndarray): Altitude of airport (unit: ft). Defaults to sea-level.
            throttle (float or ndarray): The throttle setting, between 0 and 1.
                Defaults to 1, which is at full thrust.

        Returns:
            float: Fuel flow (unit: kg/s).

        """
        Tmax = self.thrust.takeoff(tas=tas, alt=alt)
        fuelflow = throttle * self.at_thrust(Tmax)
        return fuelflow

    # @ndarrayconvert
    def enroute(self, mass, tas, alt, path_angle=0):
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

        # Convert angles from degrees to radians.
        gamma = np.radians(path_angle)

        T = D + mass * aero.g0 * np.sin(gamma)
        T_idle = self.thrust.descent_idle(tas=tas, alt=alt)
        T = np.where(T < 0, T_idle, T)

        fuelflow = self.at_thrust(T, alt)

        # do not return value outside performance boundary, with a margin of 15%
        T_max = self.thrust.climb(tas=tas, alt=alt, roc=0)
        fuelflow = np.where(T > 1.15 * T_max, np.nan, fuelflow)

        return fuelflow

    def plot_model(self, plot=True):
        """Plot the engine fuel model, or return the pyplot object.

        Args:
            plot (bool): Display the plot or return an object.

        Returns:
            None or pyplot object.

        """
        import matplotlib.pyplot as plt

        xx = np.linspace(0, 1, 50)
        yy = self.fuel_flow_model(xx)
        # plt.scatter(self.x, self.y, color='k')
        plt.plot(xx, yy, "--", color="gray")
        if plot:
            plt.show()
        else:
            return plt
