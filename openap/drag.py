"""OpenAP drag model, computed with the Aerox geometry drag model.

The clean drag comes from Aerox (https://github.com/open-aviation/aerox): a
component drag build-up from the aircraft geometry, with Mach and Reynolds
number effects and wave drag. Aerox gives the model as a closed-form
expression, so it runs with the NumPy, CasADi, and JAX backends.

``nonclean()`` adds the Aerox flap and landing gear drag: flap profile drag
(McCormick), a lower span efficiency with flaps (Roskam), and gear drag
(Torenbeek).
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from aerox import load_aircraft
from aerox.models.drag import (
    TransonicGeometryDragModel,
    build_drag_model,
    build_parabolic_polar,
)
from aerox.models.drag.expression import isa_reynolds_per_m

from . import prop
from .backends import BackendType
from .base import DragBase
from .extra import ndarrayconvert

# Type alias for numeric inputs (scalar, array, or symbolic)
Numeric = Any

# Lower bound of the Reynolds number per metre, which keeps the skin-friction
# logarithm defined at very low speed.
_MIN_REYNOLDS_PER_M = 1.0e5


class _SmoothMath:
    """Backend functions with a smooth maximum, for gradient-based solvers."""

    def __init__(self, backend):
        self.log = backend.log
        self.sqrt = backend.sqrt
        self.sin = backend.sin
        self.exp = backend.exp
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
            use_synonym: Has no effect. Kept for compatibility.
        """
        super().__init__(ac, backend=backend, **kwargs)

        self.aircraft = prop.aircraft(self.ac)
        aerox_aircraft = load_aircraft(self.ac.upper())
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

        self.polar = self._polar(aerox_aircraft)

        b = self.backend
        self._math = _SmoothMath(b) if getattr(b, "smooth_guards", False) else b

    def _polar(self, aerox_aircraft) -> Dict[str, Any]:
        """Parabolic polar at the Aerox reference condition."""
        polar = build_parabolic_polar(aerox_aircraft)
        return {"clean": {"cd0": polar.cd0, "k": polar.k}}

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
        """Mach number and Reynolds number per metre in the ISA atmosphere."""
        h = alt * self.aero.ft
        mach = self.aero.tas2mach(tas * self.aero.kts, h, dT=dT)
        reynolds = isa_reynolds_per_m(mach, h, dT, xp=self._math)
        return mach, self._math.maximum(reynolds, _MIN_REYNOLDS_PER_M)

    def _drag(
        self,
        mass: Numeric,
        tas: Numeric,
        alt: Numeric,
        vs: Numeric = 0,
        dT: Numeric = 0,
        **configuration,
    ) -> Numeric:
        cl, qS = self._cl(mass, tas, alt, vs, dT=dT)
        mach, reynolds = self._mach_reynolds(tas, alt, dT=dT)
        cd = self.expression.cd(cl, mach, reynolds, xp=self._math, **configuration)
        return cd * qS

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
        return self._drag(mass, tas, alt, vs, dT=dT)

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

        The Aerox drag with flaps and, optionally, the landing gear deployed.

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
        return self._drag(
            mass,
            tas,
            alt,
            vs,
            dT=dT,
            flap_angle_rad=flap_angle * self.backend.pi / 180,
            gear_extension_fraction=1.0 if landing_gear else 0.0,
        )
