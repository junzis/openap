"""OpenAP drag model."""

import glob
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from . import prop
from .backends import BackendType

# Type alias for numeric inputs (scalar, array, or symbolic)
Numeric = Any
from .base import DragBase
from .extra import ndarrayconvert

warnings.simplefilter("once", UserWarning)


class Drag(DragBase):
    """Compute the drag of an aircraft."""

    def __init__(
        self,
        ac: str,
        wave_drag: bool = False,
        backend: Optional[BackendType] = None,
        **kwargs,
    ):
        """Initialize Drag object.

        Args:
            ac: ICAO aircraft type (for example: A320).
            wave_drag: Enable wave drag model (experimental).
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        super().__init__(ac, backend=backend, **kwargs)

        self.use_synonym = kwargs.get("use_synonym", False)

        self.aircraft = prop.aircraft(self.ac, **kwargs)
        self.polar = self.load_drag_model()

        self.wave_drag = wave_drag
        if self.wave_drag:
            warnings.warn("Warning: Wave drag is experimental.")

    def load_drag_model(self) -> Dict[str, Any]:
        """Find and construct the drag polar model.

        Returns:
            dict: Drag polar model parameters.
        """
        # Load drag polar data
        curr_path = os.path.dirname(os.path.realpath(__file__))
        dir_dragpolar = os.path.join(curr_path, "data/dragpolar/")
        file_synonym = os.path.join(curr_path, "data/dragpolar/_synonym.csv")
        polar_synonym = pd.read_csv(file_synonym)

        polar_files = glob.glob(dir_dragpolar + "*.yml")
        ac_polar_available = [s[-8:-4].lower() for s in polar_files]

        if self.ac in ac_polar_available:
            ac = self.ac
        else:
            syno = polar_synonym.query("orig==@self.ac")
            if self.use_synonym and syno.shape[0] > 0:
                ac = syno.new.iloc[0]
                warnings.warn(
                    f"Drag polar: using synonym {ac} for {self.ac}",
                    UserWarning,
                    stacklevel=0,
                )
            elif self.use_synonym:
                raise ValueError(
                    f"Drag polar for {self.ac} not available, and no synonym found."
                )
            else:
                raise ValueError(
                    f"Drag polar for {self.ac} not available. "
                    "Try to set `use_synonym=True` to initialize the object."
                )

        f = dir_dragpolar + ac + ".yml"
        with open(f, "r") as file:
            dragpolar = yaml.safe_load(file.read())
        return dragpolar

    @ndarrayconvert
    def _cl(
        self,
        mass: Numeric,
        tas: Numeric,
        alt: Numeric,
        vs: Numeric = 0,
        dT: Numeric = 0,
    ) -> Tuple[Numeric, Numeric]:
        """Compute lift coefficient and dynamic pressure.

        Args:
            mass: Aircraft mass (kg).
            tas: True airspeed (kt).
            alt: Altitude (ft).
            vs: Vertical speed (ft/min). Defaults to 0.
            dT: Temperature deviation (K). Defaults to 0.

        Returns:
            Tuple of (lift coefficient, dynamic pressure * area).
        """
        b = self.backend

        v = tas * self.aero.kts
        h = alt * self.aero.ft
        vs = vs * self.aero.fpm
        gamma = b.arctan2(vs, v)
        S = self.aircraft["wing"]["area"]
        rho = self.aero.density(h, dT=dT)
        qS = 0.5 * rho * v**2 * S
        L = mass * self.aero.g0 * b.cos(gamma)
        qS = b.maximum(qS, 1e-3)  # avoid zero division
        cl = L / qS

        return cl, qS

    @ndarrayconvert
    def _calc_drag(
        self,
        mass: Numeric,
        tas: Numeric,
        alt: Numeric,
        cd0: Numeric,
        k: Numeric,
        vs: Numeric,
        dT: Numeric = 0,
    ) -> Numeric:
        """Compute drag from drag polar coefficients.

        Args:
            mass: Aircraft mass (kg).
            tas: True airspeed (kt).
            alt: Altitude (ft).
            cd0: Zero-lift drag coefficient.
            k: Induced drag factor.
            vs: Vertical speed (ft/min).
            dT: Temperature deviation (K). Defaults to 0.

        Returns:
            Total drag (N).
        """
        cl, qS = self._cl(mass, tas, alt, vs, dT=dT)
        cd = cd0 + k * cl**2
        D = cd * qS
        return D

    @ndarrayconvert
    def clean(
        self,
        mass: Numeric,
        tas: Numeric,
        alt: Numeric,
        vs: Numeric = 0,
        dT: Numeric = 0,
    ) -> Numeric:
        """Compute drag at clean configuration (considering compressibility).

        Args:
            mass: Mass of the aircraft (kg).
            tas: True airspeed (kt).
            alt: Altitude (ft).
            vs: Vertical rate (ft/min). Defaults to 0.
            dT: Temperature shift (K or degC). Defaults to 0.

        Returns:
            Total drag (N).
        """
        b = self.backend

        cd0 = self.polar["clean"]["cd0"]
        k = self.polar["clean"]["k"]

        if self.wave_drag:
            mach = self.aero.tas2mach(tas * self.aero.kts, alt * self.aero.ft, dT=dT)
            cl, qS = self._cl(mass, tas, alt, dT=dT)

            sweep = self.aircraft["wing"]["sweep"] * b.pi / 180
            tc = self.aircraft["wing"]["t/c"]

            # Default thickness to chord ratio, based on Obert (2009)
            if tc is None:
                tc = 0.12

            cos_sweep = b.cos(sweep)

            kappa = 0.95  # assume supercritical airfoils

            # Equation 17 and 18 in Gur et al. (2010) - for conventional airfoil
            mach_crit = (
                kappa / cos_sweep - tc / cos_sweep**2 - 0.1 * cl / cos_sweep**3 - 0.108
            )

            # Equation 15 in Gur et al. (2010)
            dmach = b.maximum(mach - mach_crit, 0.0)
            dCdw = 20 * dmach**4

        else:
            dCdw = 0

        cd0 = cd0 + dCdw

        D = self._calc_drag(mass, tas, alt, cd0, k, vs, dT=dT)
        return D

    @ndarrayconvert
    def nonclean(
        self,
        mass: Numeric,
        tas: Numeric,
        alt: Numeric,
        flap_angle: Numeric,
        vs: Numeric = 0,
        dT: Numeric = 0,
        landing_gear: bool = False,
    ) -> Numeric:
        """Compute drag at non-clean configuration.

        Args:
            mass: Mass of the aircraft (kg).
            tas: True airspeed (kt).
            alt: Altitude (ft).
            flap_angle: Flap deflection angle (degree).
            vs: Vertical rate (ft/min). Defaults to 0.
            dT: Temperature shift (K or degC). Defaults to 0.
            landing_gear: Is landing gear extended? Defaults to False.

        Returns:
            Total drag (N).
        """
        b = self.backend

        cd0 = self.polar["clean"]["cd0"]
        k = self.polar["clean"]["k"]

        # --- calc new CD0 ---
        lambda_f = self.polar["flaps"]["lambda_f"]
        cfc = self.polar["flaps"]["cf/c"]
        SfS = self.polar["flaps"]["Sf/S"]

        # Equation 3.45-3.46 in McCormick (1994), page 109.
        delta_cd_flap = (
            lambda_f * (cfc) ** 1.38 * (SfS) * b.sin(flap_angle * b.pi / 180) ** 2
        )

        if landing_gear:
            # Equation 6.1 in Mair and Birdsall (1996)
            # 3.16e-5 is the K_uc factor value corresponding to
            # maximum flap deflection
            delta_cd_gear = (
                self.aircraft["limits"]["MTOW"]
                * self.aero.g0
                / self.aircraft["wing"]["area"]
                * 3.16e-5
                * self.aircraft["limits"]["MTOW"] ** (-0.215)
            )
        else:
            delta_cd_gear = 0

        cd0_total = cd0 + delta_cd_flap + delta_cd_gear

        # --- calc new k ---
        if self.aircraft["engine"]["mount"] == "rear":
            # See Figure 27.38 in Obert (2009)
            delta_e_flap = 0.0046 * flap_angle
        else:
            # See Figure 27.39 in Obert (2009)
            delta_e_flap = 0.0026 * flap_angle

        ar = self.aircraft["wing"]["span"] ** 2 / self.aircraft["wing"]["area"]
        k_total = 1 / (1 / k + b.pi * ar * delta_e_flap)

        D = self._calc_drag(mass, tas, alt, cd0_total, k_total, vs, dT=dT)
        return D
