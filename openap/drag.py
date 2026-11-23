"""OpenAP drag model, computed with the Aerox geometry drag model.

The clean drag comes from Aerox (https://github.com/open-aviation/aerox): a
component drag build-up from the aircraft geometry, with Mach and Reynolds
number effects and wave drag. Aerox gives the model as a closed-form
expression, so it runs with the NumPy, CasADi, and JAX backends.

Aerox covers the clean configuration only. ``nonclean()`` adds OpenAP's flap
and landing gear increments to the Aerox clean drag.
"""

import os
import warnings
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import yaml
from aerox import get_available_aircraft, load_aircraft
from aerox.models.drag import (
    TransonicGeometryDragModel,
    build_drag_model,
    build_parabolic_polar,
)

import pandas as pd

from . import prop
from .backends import BackendType
from .base import DragBase
from .extra import ndarrayconvert

# Type alias for numeric inputs (scalar, array, or symbolic)
Numeric = Any

warnings.simplefilter("once", UserWarning)

_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

# Flap parameters for aircraft without OpenAP flap data (the most common values)
_DEFAULT_FLAPS = {"lambda_f": 0.9, "cf/c": 0.15, "Sf/S": 0.15}

# Lower bound of the Reynolds number per metre, which keeps the skin-friction
# logarithm defined at very low speed.
_MIN_REYNOLDS_PER_M = 1.0e5


class _SmoothMath:
    """Backend functions with a smooth maximum, for gradient-based solvers."""

    def __init__(self, backend):
        self.log = backend.log
        self.sqrt = backend.sqrt
        self._backend = backend

    def maximum(self, x, y):
        return self._backend.smooth_max(x, y, softness=1e-3)


class Drag(DragBase):
    """Compute the drag of an aircraft."""

    def __init__(
        self,
        ac: str,
        wave_drag: bool = True,
        backend: Optional[BackendType] = None,
        **kwargs,
    ):
        """Initialize Drag object.

        Args:
            ac: ICAO aircraft type (for example: A320).
            wave_drag: Include wave drag. Defaults to True.
            backend: Math backend to use. Defaults to NumpyBackend.
            use_synonym: Use a similar aircraft type if this one is not
                available. Defaults to False.
        """
        super().__init__(ac, backend=backend, **kwargs)

        self.use_synonym = kwargs.get("use_synonym", False)

        aerox_aircraft = self._load_aerox_aircraft()
        self.model = build_drag_model(aerox_aircraft)
        geometry = (
            self.model.base_model
            if isinstance(self.model, TransonicGeometryDragModel)
            else self.model
        )
        self.wing_area = geometry.reference_area_m2
        self.wing_span = geometry.wing_span_m

        expression = self.model.expression()
        if not wave_drag:
            expression = replace(
                expression, critical_mach_zero_lift=None, critical_mach_slope=None
            )
        self.expression = expression
        self.wave_drag = expression.has_wave_drag

        self.aircraft = self._aircraft_properties(aerox_aircraft, **kwargs)
        self.polar = self._polar(aerox_aircraft)

        b = self.backend
        self._math = _SmoothMath(b) if getattr(b, "smooth_guards", False) else b

    def _load_aerox_aircraft(self):
        available = set(get_available_aircraft())
        code = self.ac.upper()
        if code in available:
            return load_aircraft(code)

        synonyms = pd.read_csv(os.path.join(_DATA, "dragpolar/_synonym.csv"))
        match = synonyms.query("orig==@self.ac")
        if self.use_synonym and match.shape[0] > 0:
            new = match.new.iloc[0].upper()
            if new in available:
                warnings.warn(
                    f"Drag: using synonym {new} for {code}", UserWarning, stacklevel=0
                )
                return load_aircraft(new)
        if self.use_synonym:
            raise ValueError(
                f"Drag model for {code} not available, and no synonym found."
            )
        raise ValueError(
            f"Drag model for {code} not available. "
            "Try to set `use_synonym=True` to initialize the object."
        )

    def _aircraft_properties(self, aerox_aircraft, **kwargs) -> Dict[str, Any]:
        """OpenAP aircraft properties, or the Aerox record for other types."""
        try:
            return prop.aircraft(self.ac, **kwargs)
        except ValueError:
            pass
        mount = str(aerox_aircraft.engine.mount or "").lower()
        mtow = aerox_aircraft.mtow
        return {
            "aircraft": aerox_aircraft.aircraft,
            "mtow": mtow,
            "limits": {"MTOW": mtow},
            "wing": {"area": self.wing_area, "span": self.wing_span},
            "engine": {"mount": "rear" if "rear" in mount or "aft" in mount else mount},
        }

    def _polar(self, aerox_aircraft) -> Dict[str, Any]:
        """Parabolic polar at the Aerox reference condition, and flap data."""
        polar = build_parabolic_polar(aerox_aircraft)
        flaps = dict(_DEFAULT_FLAPS)
        path = os.path.join(_DATA, "dragpolar", f"{self.ac}.yml")
        if os.path.exists(path):
            with open(path, "r") as file:
                flaps.update(yaml.safe_load(file.read()).get("flaps") or {})
        return {"clean": {"cd0": polar.cd0, "k": polar.k}, "flaps": flaps}

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
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
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
        rho = self.aero.density(h, dT=dT)
        qS = 0.5 * rho * v**2 * self.wing_area
        mass = self.mass_to_kg(mass)
        L = mass * self.aero.g0 * b.cos(gamma)
        if getattr(b, "smooth_guards", False):
            qS = b.smooth_max(qS, 1e-3, softness=1e-4)
        else:
            qS = b.maximum(qS, 1e-3)  # avoid zero division
        cl = L / qS

        return cl, qS

    def _mach_reynolds(
        self, tas: Numeric, alt: Numeric, dT: Numeric = 0
    ) -> Tuple[Numeric, Numeric]:
        """Mach number and Reynolds number per metre (Sutherland viscosity)."""
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        mach = self.aero.tas2mach(v, h, dT=dT)
        rho = self.aero.density(h, dT=dT)
        T = self.aero.temperature(h, dT=dT)
        mu = 1.458e-6 * T**1.5 / (T + 110.4)
        reynolds = self._math.maximum(rho * v / mu, _MIN_REYNOLDS_PER_M)
        return mach, reynolds

    def _clean_cd(
        self,
        mass: Numeric,
        tas: Numeric,
        alt: Numeric,
        vs: Numeric = 0,
        dT: Numeric = 0,
    ) -> Tuple[Numeric, Numeric, Numeric]:
        cl, qS = self._cl(mass, tas, alt, vs, dT=dT)
        mach, reynolds = self._mach_reynolds(tas, alt, dT=dT)
        cd = self.expression.cd(cl, mach, reynolds, xp=self._math)
        return cd, cl, qS

    @ndarrayconvert
    def clean(
        self,
        mass: Numeric,
        tas: Numeric,
        alt: Numeric,
        vs: Numeric = 0,
        dT: Numeric = 0,
    ) -> Numeric:
        """Compute drag at clean configuration.

        The drag coefficient depends on the lift coefficient, the Mach number,
        and the Reynolds number, and includes wave drag unless the model was
        created with ``wave_drag=False``.

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Altitude (ft).
            vs: Vertical rate (ft/min). Defaults to 0.
            dT: Temperature shift (K or degC). Defaults to 0.

        Returns:
            Total drag (N).
        """
        cd, _, qS = self._clean_cd(mass, tas, alt, vs, dT=dT)
        return cd * qS

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

        The Aerox clean drag plus OpenAP's flap and landing gear increments.

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
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

        cd_clean, cl, qS = self._clean_cd(mass, tas, alt, vs, dT=dT)

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
            mtow = self.aircraft["limits"]["MTOW"]
            delta_cd_gear = (
                mtow * self.aero.g0 / self.wing_area * 3.16e-5 * mtow ** (-0.215)
            )
        else:
            delta_cd_gear = 0

        # Induced drag factor with flaps
        if self.aircraft["engine"]["mount"] == "rear":
            # See Figure 27.38 in Obert (2009)
            delta_e_flap = 0.0046 * flap_angle
        else:
            # See Figure 27.39 in Obert (2009)
            delta_e_flap = 0.0026 * flap_angle

        k = self.expression.span_induced_factor
        ar = self.wing_span**2 / self.wing_area
        k_flap = 1 / (1 / k + b.pi * ar * delta_e_flap)

        cd = cd_clean + delta_cd_flap + delta_cd_gear + (k_flap - k) * cl**2
        return cd * qS
