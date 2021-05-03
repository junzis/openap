"""OpenAP drag model."""

import os
import glob
import yaml
import warnings
import numpy as np
from openap import prop
from openap.extra import aero
from openap.extra import ndarrayconvert

import casadi as ca

curr_path = os.path.dirname(os.path.realpath(__file__))
dir_dragpolar = curr_path + "/data/dragpolar/"


class Drag(object):
    """Compute the drag of aicraft."""

    def __init__(self, ac, wave_drag=False):
        """Initialize Drag object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).

        """
        super(Drag, self).__init__()

        self.ac = ac.lower()
        self.aircraft = prop.aircraft(ac)
        self.polar = self.dragpolar()
        self.wave_drag = wave_drag

        if self.wave_drag:
            warnings.warn(
                "Performance warning: Wave drag model is inaccurate at the moment. This will be fixed in future release."
            )

    def dragpolar(self):
        """Find and construct the drag polar model.

        Returns:
            dict: drag polar model parameters.

        """
        polar_files = glob.glob(dir_dragpolar + "*.yml")
        ac_polar_available = [s[-8:-4].lower() for s in polar_files]

        if self.ac in ac_polar_available:
            ac = self.ac
        else:
            if self.ac.startswith("a32"):
                ac = "a320"
            elif self.ac.startswith("a33"):
                ac = "a332"
            elif self.ac.startswith("a34"):
                ac = "a333"
            elif self.ac.startswith("a35"):
                ac = "a359"
            elif self.ac.startswith("a38"):
                ac = "a388"
            elif self.ac.startswith("b73"):
                ac = "b738"
            elif self.ac.startswith("b74"):
                ac = "b744"
            elif self.ac.startswith("b77"):
                ac = "b77w"
            else:
                raise RuntimeError("%s drag polar not available." % self.ac.upper())

            print("warning: %s drag polar used for %s." % (ac.upper(), self.ac.upper()))

        f = dir_dragpolar + ac + ".yml"
        dragpolar = yaml.safe_load(open(f))
        return dragpolar

    # @ndarrayconvert
    def _calc_drag(self, mass, tas, alt, cd0, k, path_angle):
        v = tas * aero.kts
        h = alt * aero.ft
        # gamma = np.radians(path_angle)
        gamma = path_angle*np.pi/180

        S = self.aircraft["wing"]["area"]

        rho = aero.density(h)
        qS = 0.5 * rho * v ** 2 * S
        L = mass * aero.g0 * np.cos(gamma)
        qS = ca.if_else(qS<1e-3,1e-3,qS)
        # qS = np.where(qS < 1e-3, 1e-3, qS)
        cl = L / qS
        cd = cd0 + k * cl ** 2
        D = cd * qS
        
        # D = int(D)

        # D = D.astype(int)

        return D

    # @ndarrayconvert
    def clean(self, mass, tas, alt, path_angle=0):
        """Compute drag at clean configuration (considering compressibility).

        Args:
            mass (int or ndarray): Mass of the aircraft (unit: kg).
            tas (int or ndarray): True airspeed (unit: kt).
            alt (int or ndarray): Altitude (unit: ft).
            path_angle (float or ndarray): Path angle (unit: degree). Defaults to 0.

        Returns:
            int: Total drag (unit: N).

        """

        cd0 = self.polar["clean"]["cd0"]
        k = self.polar["clean"]["k"]

        if self.wave_drag:
            mach_crit = self.polar["mach_crit"]
            mach = aero.tas2mach(tas * aero.kts, alt * aero.ft)

            dCdw = ca.if_else(mach > mach_crit, 20 * (mach - mach_crit) ** 4, 0)
        else:
            dCdw = 0

        cd0 = cd0 + dCdw

        D = self._calc_drag(mass, tas, alt, cd0, k, path_angle)
        return D

    # @ndarrayconvert
    def nonclean(self, mass, tas, alt, flap_angle, path_angle=0, landing_gear=False):
        """Compute drag at at non-clean configuratio.

        Args:
            mass (int or ndarray): Mass of the aircraft (unit: kg).
            tas (int or ndarray): True airspeed (unit: kt).
            alt (int or ndarray): Altitude (unit: ft).
            flap_angle (int or ndarray): flap deflection angle (unit: degree).
            path_angle (float or ndarray): Path angle (unit: degree). Defaults to 0.
            landing_gear (bool): Is landing gear extended? Defaults to False.

        Returns:
            int or ndarray: Total drag (unit: N).

        """
        cd0 = self.polar["clean"]["cd0"]
        k = self.polar["clean"]["k"]

        # --- calc new CD0 ---
        lambda_f = self.polar["flaps"]["lambda_f"]
        cfc = self.polar["flaps"]["cf/c"]
        SfS = self.polar["flaps"]["Sf/S"]

        delta_cd_flap = (
            lambda_f * (cfc) ** 1.38 * (SfS) * np.sin(np.radians(flap_angle)) ** 2
        )

        if landing_gear:
            delta_cd_gear = (
                self.aircraft["limits"]["MTOW"]
                * 9.8065
                / self.aircraft["wing"]["area"]
                * 3.16e-5
                * self.aircraft["limits"]["MTOW"] ** (-0.215)
            )
        else:
            delta_cd_gear = 0

        cd0_total = cd0 + delta_cd_flap + delta_cd_gear

        # --- calc new k ---
        if self.aircraft["engine"]["mount"] == "rear":
            delta_e_flap = 0.0046 * flap_angle
        else:
            delta_e_flap = 0.0026 * flap_angle

        ar = self.aircraft["wing"]["span"] ** 2 / self.aircraft["wing"]["area"]
        k_total = 1 / (1 / k + np.pi * ar * delta_e_flap)

        D = self._calc_drag(mass, tas, alt, cd0_total, k_total, path_angle)
        return D
