# %%
from glob import glob
from xml.etree import ElementTree

from numpy import ndarray

from .. import base
from ..extra import ndarrayconvert


# %%
def _poly1(coefficients, x, exponents):
    total = 0.0
    for coefficient, exponent in zip(coefficients, exponents):
        total = total + coefficient * x**exponent
    return total


def _poly2(coefficients, shape, row_terms, col_terms):
    rows, cols = shape
    total = 0.0
    for i in range(rows):
        for j in range(cols):
            total = total + coefficients[i * cols + j] * row_terms[i] * col_terms[j]
    return total


def load_bada4(ac: str, path: str) -> ElementTree.ElementTree:
    """Load BADA4 model XML.

    Args:
        ac: Aircraft type (for example: A320 or A320-231).
        path: Path to BADA4 models.

    Returns:
        BADA4 model XML tree.

    """

    ac_options = glob(f"{path}/{ac.upper()}*")
    if not ac_options:
        raise ValueError(f"No BADA4 model found for {ac}.")

    model_path = ac_options[0]
    model_xml_path = glob(f"{model_path}/*.xml")[0]

    badatree = ElementTree.parse(model_xml_path)
    return badatree


# %%
class Drag(base.DragBase):
    """Compute the drag of an aircraft using BADA4 models."""

    def __init__(self, ac: str, bada_path: str, **kwargs):
        """Initialize Drag object.

        Args:
            ac: Aircraft type (for example: A320).
            bada_path: Path to BADA4 models.

        """
        super().__init__(ac, **kwargs)

        self.ac = ac.upper()
        self.smooth = kwargs.get("smooth", False)

        # load parameters from xml
        bxml = load_bada4(ac, bada_path)
        self.scalar = float(bxml.findtext(".//*/DPM_clean/scalar"))
        self.d_ = [float(v.text) for v in bxml.findall(".//*/CD_clean/d")]
        self.mach_max = float(bxml.findtext(".//*/DPM_clean/M_max"))
        self.S = float(bxml.findtext("./AFCM/S"))

    @ndarrayconvert(column=True)
    def _cd_base(self, cl, mach):
        mm = (1 - mach**2) ** (-0.5)

        C0 = _poly1(self.d_[0:5], mm, range(5))
        C2 = _poly1(self.d_[5:10], mm, range(0, 13, 3))
        C6 = self.d_[10] + _poly1(self.d_[11:15], mm, range(14, 18))

        cd = self.scalar * (C0 + C2 * cl**2 + C6 * cl**6)

        return cd

    def _cd_params_base(self, mach):
        mm = (1 - mach**2) ** (-0.5)
        return {
            "cd0": self.scalar * _poly1(self.d_[0:5], mm, range(5)),
            "cd2": self.scalar * _poly1(self.d_[5:10], mm, range(0, 13, 3)),
            "cd6": self.scalar
            * (self.d_[10] + _poly1(self.d_[11:15], mm, range(14, 18))),
        }

    def clean_drag_polar_params(self, tas, alt, dT=0):
        """Return clean-configuration drag polar parameters.

        BADA4 clean drag is CD = cd0(M) + cd2(M) * CL**2 + cd6(M) * CL**6.
        The returned parameters include the same Mach-divergence correction
        used by :meth:`clean`.
        """
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        mach = self.aero.tas2mach(v, h, dT=dT)

        mach_base = self.mach_max - 0.01
        if self.smooth:
            divergent = self.backend.smooth_max(
                (mach - mach_base) / 0.01,
                0,
                softness=1e-3,
            )
        else:
            divergent = self.backend.maximum((mach - mach_base) / 0.01, 0)

        base = self._cd_params_base(mach)
        at_base = self._cd_params_base(mach_base)
        at_max = self._cd_params_base(self.mach_max)

        params = {}
        for name in ("cd0", "cd2", "cd6"):
            divergent_value = at_base[name] + divergent**1.5 * (
                at_max[name] - at_base[name]
            )
            if self.smooth:
                params[name] = self.backend.smooth_switch(
                    mach,
                    self.mach_max,
                    base[name],
                    divergent_value,
                    softness=1e-3,
                )
            else:
                params[name] = self.backend.where(
                    mach < self.mach_max, base[name], divergent_value
                )
        return params

    @ndarrayconvert(column=True)
    def _cd(self, cl, mach):
        """Compute the drag coefficient (CD)"""

        cd = self._cd_base(cl, mach)

        # when M > M_max
        mach_base = self.mach_max - 0.01
        cd_mach_max = self._cd_base(cl, self.mach_max)
        cd_mach_base = self._cd_base(cl, mach_base)

        divergent = (mach - mach_base) / 0.01
        if self.smooth:
            divergent = self.backend.smooth_max(divergent, 0, softness=1e-3)
        else:
            divergent = self.backend.maximum(divergent, 0)
        cd_crit = cd_mach_base + divergent**1.5 * (cd_mach_max - cd_mach_base)

        if self.smooth:
            cd = self.backend.smooth_switch(
                mach,
                self.mach_max,
                cd,
                cd_crit,
                softness=1e-3,
            )
        else:
            cd = self.backend.where(mach < self.mach_max, cd, cd_crit)

        return cd

    @ndarrayconvert(column=True)
    def _cl(self, mass, tas, alt, vs=0, dT=0):
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        rho = self.aero.density(h, dT=dT)

        qS = 0.5 * rho * v**2 * self.S
        mass = self.mass_to_kg(mass)
        L = mass * self.aero.g0

        if self.smooth:
            qS_safe = self.backend.smooth_max(qS, 1e-3, softness=1e-4)
        else:
            qS_safe = self.backend.maximum(qS, 1e-3)
        cl = L / qS_safe  # avoid zero division

        return cl, qS

    @ndarrayconvert(column=True)
    def clean(self, mass, tas, alt, vs=0, dT=0) -> float | ndarray:
        """Compute drag at clean configuration.

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Altitude (ft).
            vs: Vertical rate (ft/min). Defaults to 0.

        Returns:
            Total drag (N).

        """
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        mach = self.aero.tas2mach(v, h, dT=dT)

        cl, qS = self._cl(mass, tas, alt, vs, dT=dT)
        cd = self._cd(cl, mach)
        D = cd * qS

        return D


class Thrust(base.ThrustBase):
    """Compute the thrust of an aircraft using BADA4 models."""

    def __init__(self, ac: str, bada_path: str, **kwargs):
        """Initialize Thrust object.

        Args:
            ac: Aircraft type (for example: A320).
            bada_path: Path to BADA4 models.

        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()
        self.smooth = kwargs.get("smooth", False)

        # load parameters from xml
        bxml = load_bada4(ac, bada_path)
        self.m_ref = float(bxml.findtext("./PFM/MREF"))
        self.a_ = [float(v.text) for v in bxml.findall("./PFM/TFM/CT/a")]
        self.ti = [float(v.text) for v in bxml.findall("./PFM/TFM/LIDL/CT/ti")]

        self.kink = dict()
        self.b_ = dict()
        self.c_ = dict()

        for rating in ["MCRZ", "MCMB", "MTKF"]:
            kink = bxml.findtext(f"./PFM/TFM/{rating}/kink")
            if kink is None:
                continue
            self.kink[rating] = float(kink)
            self.b_[rating] = [
                float(t.text) for t in bxml.findall(f"./PFM/TFM/{rating}/flat_rating/b")
            ]

            self.c_[rating] = [
                float(t.text) for t in bxml.findall(f"./PFM/TFM/{rating}/temp_rating/c")
            ]

    @ndarrayconvert(column=True)
    def cT(self, mach, h, rating, dT=0) -> float | ndarray:
        """Compute the thrust coefficient.

        Args:
            mach: Mach number.
            h: Altitude (m).
            rating: Thrust rating ('MCRZ', 'MCMB', 'MTKF', or 'LIDL').
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Thrust coefficient.

        """

        rating = rating.upper()
        assert rating in ["MCRZ", "MCMB", "MTKF", "LIDL"]

        k = 1.4

        delta = self.aero.pressure(h, dT=dT) / self.aero.p0
        theta = self.aero.temperature(h, dT=dT) / self.aero.T0

        if rating == "LIDL":
            delta_terms = [delta**i for i in range(-1, 3)]
            mach_terms = [mach**i for i in range(3)]
            cT = _poly2(self.ti, (3, 4), mach_terms, delta_terms)

        else:
            b = self.backend
            mach_terms_6 = [mach**i for i in range(6)]
            delta_terms_6 = [delta**i for i in range(6)]
            flat_delta_T = _poly2(self.b_[rating], (6, 6), delta_terms_6, mach_terms_6)

            mach_terms_5 = [mach**i for i in range(5)]
            theta_t = theta * (1 + (mach**2) * (k - 1) / 2)
            temp_terms = [theta_t**i for i in range(5)] + [
                delta**i for i in range(1, 5)
            ]
            temp_delta_T = _poly2(self.c_[rating], (9, 5), temp_terms, mach_terms_5)
            if self.smooth:
                delta_T = b.smooth_switch(
                    dT,
                    self.kink[rating],
                    flat_delta_T,
                    temp_delta_T,
                    softness=0.1,
                )
            else:
                delta_T = b.where(dT <= self.kink[rating], flat_delta_T, temp_delta_T)

            delta_T_terms = [delta_T**i for i in range(6)]
            cT = _poly2(self.a_, (6, 6), delta_T_terms, mach_terms_6)

        return cT

    @ndarrayconvert(column=True)
    def climb(self, tas, alt, dT=0) -> float | ndarray:
        """Compute the thrust force during the climb phase.

        Args:
            tas: True airspeed (kt).
            alt: Altitude (ft).
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Thrust force during climb (N).

        """
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h, dT=dT)
        delta = self.aero.pressure(h, dT=dT) / self.aero.p0

        cT = self.cT(mach, h, "MCMB", dT)

        return delta * self.m_ref * self.aero.g0 * cT

    @ndarrayconvert(column=True)
    def cruise(self, tas, alt, dT=0) -> float | ndarray:
        """Compute the thrust force during the cruise phase.

        Args:
            tas: True airspeed (kt).
            alt: Altitude (ft).
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Thrust force during cruise (N).

        """
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h, dT=dT)
        delta = self.aero.pressure(h, dT=dT) / self.aero.p0

        cT = self.cT(mach, h, "MCRZ", dT)

        return delta * self.m_ref * self.aero.g0 * cT

    @ndarrayconvert(column=True)
    def takeoff(self, tas, alt=0, dT=0) -> float | ndarray:
        """Compute the thrust force at takeoff.

        Args:
            tas: True airspeed (kt).
            alt: Altitude (ft). Defaults to 0.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Thrust force during takeoff (N).

        """
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h, dT=dT)
        delta = self.aero.pressure(h, dT=dT) / self.aero.p0

        rating = "MTKF" if "MTKF" in self.kink else "MCMB"
        cT = self.cT(mach, h, rating, dT)

        return delta * self.m_ref * self.aero.g0 * cT

    @ndarrayconvert(column=True)
    def idle(self, tas, alt=0, dT=0) -> float | ndarray:
        """Compute the idle thrust.

        Args:
            tas: True airspeed (kt).
            alt: Altitude (ft). Defaults to 0.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Idle thrust force (N).

        """
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h, dT=dT)
        delta = self.aero.pressure(h, dT=dT) / self.aero.p0

        cT = self.cT(mach, h, "LIDL", dT)

        return delta * self.m_ref * self.aero.g0 * cT


# %%
class FuelFlow(base.FuelFlowBase):
    """Compute the fuel flow of an aircraft using BADA4 models."""

    def __init__(self, ac: str, bada_path: str, **kwargs):
        """Initialize FuelFlow object.

        Args:
            ac: Aircraft type (for example: A320).
            bada_path: Path to BADA4 models.

        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()
        self.smooth = kwargs.get("smooth", False)
        self.thrust = Thrust(ac, bada_path, backend=self.backend, smooth=self.smooth)
        self.drag = Drag(ac, bada_path, backend=self.backend, smooth=self.smooth)

        # load parameters from xml
        bxml = load_bada4(ac, bada_path)
        self.mass_ref = float(bxml.findtext("./PFM/MREF"))
        self.f_ = [float(v.text) for v in bxml.findall("./PFM/TFM/CF/f")]
        self.fi_ = [float(v.text) for v in bxml.findall("./PFM/TFM/LIDL/CF/fi")]
        self.lhv = float(bxml.findtext("./PFM/LHV"))

    @ndarrayconvert(column=True)
    def _calc_fuel(self, mass, delta, theta, cF):
        return (
            delta
            * (theta**0.5)
            * self.mass_ref
            * self.aero.g0
            * self.aero.a0
            * (1 / self.lhv)
            * cF
        )

    @ndarrayconvert(column=True)
    def idle(self, mass, tas, alt, **kwargs) -> float | ndarray:
        """Compute the fuel flow at idle conditions.

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: Aircraft true airspeed (kt).
            alt: Aircraft altitude (ft).
            dT: Temperature deviation (K). Defaults to 0.

        Returns:
            Fuel flow (kg/s).

        """

        mass = self.mass_to_kg(mass)
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        dT = kwargs.get("dT", 0)

        mach = self.aero.tas2mach(v, h, dT=dT)
        delta = self.aero.pressure(h, dT=dT) / self.aero.p0
        theta = self.aero.temperature(h, dT=dT) / self.aero.T0

        delta_terms = [delta**i for i in range(3)]
        mach_terms = [mach**i for i in range(3)]
        cF_idle = _poly2(self.fi_, (3, 3), mach_terms, delta_terms)
        cF_idle = cF_idle * delta**-1 * theta**-0.5

        fuel_flow = self._calc_fuel(mass, delta, theta, cF_idle)

        return fuel_flow

    @ndarrayconvert(column=True)
    def enroute(self, mass, tas, alt, vs=0, **kwargs) -> float | ndarray:
        """Compute the fuel flow at non-idle conditions.

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: Aircraft true airspeed (kt).
            alt: Aircraft altitude (ft).
            vs: Vertical rate (ft/min). Defaults to 0.
            dT: Temperature deviation (K). Defaults to 0.

        Returns:
            Fuel flow (kg/s).

        """
        mass = self.mass_to_kg(mass)
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        dT = kwargs.get("dT", 0)

        mach = self.aero.tas2mach(v, h, dT=dT)
        delta = self.aero.pressure(h, dT=dT) / self.aero.p0
        theta = self.aero.temperature(h, dT=dT) / self.aero.T0
        gamma = self.backend.arctan2(vs * self.aero.fpm, v)

        D = self.drag.clean(mass, tas, alt, vs, dT=dT)
        T = D + mass * self.aero.g0 * self.backend.sin(gamma)

        cT = T / (delta * self.mass_ref * self.aero.g0)

        cT_terms = [cT**i for i in range(5)]
        mach_terms = [mach**i for i in range(5)]
        cF_gen = _poly2(self.f_, (5, 5), mach_terms, cT_terms)

        fuel_flow_non_idle = self._calc_fuel(mass, delta, theta, cF_gen)
        fuel_flow_idle = self.idle(mass, tas, alt, dT=dT)

        # BADA 4 eq. 3.4-2: enroute fuel follows required thrust, with an
        # idle lower bound. A descent can still require thrust above idle.
        if self.smooth:
            fuel_flow = self.backend.smooth_max(
                fuel_flow_non_idle, fuel_flow_idle, softness=1e-3
            )
        else:
            fuel_flow = self.backend.maximum(fuel_flow_non_idle, fuel_flow_idle)

        return fuel_flow
