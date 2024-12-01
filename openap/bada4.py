# %%
from glob import glob
from xml.etree import ElementTree

from . import drag, fuel, thrust
from .extra import ndarrayconvert


# %%
def load_bada4(ac: str, path: str) -> ElementTree:
    """Find and construct the drag polar model.
    Args:
        ac (string): ICAO aircraft type (for example: A320 or A320-231).
        bada_path (string): path to BADA4 models.

    Returns:
        xml.etree.ElementTree: BADA4 model XML.
    """

    ac_options = glob(f"{path}/{ac.upper()}*")
    if not ac_options:
        raise ValueError(f"No BADA4 model found for {ac}.")

    model_path = ac_options[0]
    model_xml_path = glob(f"{model_path}/*.xml")[0]

    badatree = ElementTree.parse(model_xml_path)
    return badatree


# %%
class Drag(drag.DragBase):
    """Compute the drag of an aircraft with BADA4 models."""

    def __init__(self, ac, bada_path, **kwargs):
        """Initialize Drag object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            path (string): path to BADA4 models.

        """
        super().__init__(ac, **kwargs)

        self.ac = ac.upper()

        # load parameters from xml
        bxml = load_bada4(ac, bada_path)
        self.scalar = float(bxml.findtext(".//*/DPM_clean/scalar"))
        self.d_ = self.sci.array(
            [float(v.text) for v in bxml.findall(".//*/CD_clean/d")]
        )
        self.mach_max = float(bxml.findtext(".//*/DPM_clean/M_max"))
        self.S = float(bxml.findtext("./AFCM/S"))

    @ndarrayconvert
    def _cd_base(self, cl, mach):
        mm = (1 - mach**2) ** (-0.5)

        C0 = self.sci.dot(
            self.sci.array([mm[:, 0] ** i for i in range(5)]).T,
            self.d_[0:5].reshape(5, 1),
        )

        C2 = self.sci.dot(
            self.sci.array([mm[:, 0] ** i for i in range(0, 13, 3)]).T,
            self.d_[5:10].reshape(5, 1),
        )

        C6 = self.d_[10] + self.sci.dot(
            self.sci.array([mm[:, 0] ** i for i in range(14, 18)]).T,
            self.d_[11:15].reshape(4, 1),
        )

        cd = self.scalar * (C0 + C2 * cl**2 + C6 * cl**6)

        return cd

    @ndarrayconvert
    def _cd(self, cl, mach):
        """
        Compute the drag coefficient (CD) based on given inputs.

        Parameters:
        - CL: Lift coefficient (float or array)
        - M: Mach number (float or array)
        - scalar: Scaling factor (float)
        - d: List or array of coefficients [d1, d2, ..., d15] (length 15)

        Returns:
        - CD: Drag coefficient (float or array)
        """
        cd = self._cd_base(cl, mach)

        # when M > M_max
        mach_base = self.mach_max - 0.01
        cd_mach_max = self._cd_base(cl, self.mach_max)
        cd_mach_base = self._cd_base(cl, mach_base)

        divergent = (mach - mach_base) / 0.01
        divergent = self.sci.maximum(divergent, 0)
        cd_crit = cd_mach_base + divergent**1.5 * (cd_mach_max - cd_mach_base)

        cd = self.sci.where(mach < self.mach_max, cd, cd_crit)

        return cd

    @ndarrayconvert
    def _cl(self, mass, tas, alt, vs=0):
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        rho = self.aero.density(h)

        qS = 0.5 * rho * v**2 * self.S
        L = mass * self.aero.g0

        cl = L / self.sci.maximum(qS, 1e-3)  # avoid zero division

        return cl, qS

    @ndarrayconvert
    def clean(self, mass, tas, alt, vs=0):
        """Compute drag at clean configuration.

        Args:
            mass (int): Mass of the aircraft (unit: kg).
            tas (int): True airspeed (unit: kt).
            alt (int): Altitude (unit: ft).
            vs (float): Vertical rate (unit: feet/min). Defaults to 0.

        Returns:
            int: Total drag (unit: N).

        """
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        mach = self.aero.tas2mach(v, h)

        cl, qS = self._cl(mass, tas, alt, vs)
        cd = self._cd(cl, mach)
        D = cd * qS

        return D


class Thrust(thrust.ThrustBase):
    """Compute the thrust of an aircraft with BADA4 models."""

    def __init__(self, ac, bada_path, **kwargs):
        """Initialize Thrust object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            path (string): path to BADA4 models.

        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()

        # load parameters from xml
        bxml = load_bada4(ac, bada_path)
        self.a_ = [float(v.text) for v in bxml.findall("./PFM/TFM/CT/a")]
        self.b_ = dict()
        self.c_ = dict()
        for rating in ["MCRZ", "MCMB", "LIDL"]:
            self.b_[rating] = [
                float(t.text) for t in bxml.findall(f".//*/{rating}/flat_rating/b")
            ]

            self.c_[rating] = [
                float(t.text) for t in bxml.findall(f".//*/{rating}/temp_rating/c")
            ]

    @ndarrayconvert
    def cT(self, mach, alt, rating, dT=0):
        """Compute the thrust coefficient, considering the altitude, Mach and phase."""

        rating = rating.upper()
        assert rating in ["MCRZ", "MCMB", "LIDL"]

        h = alt * self.aero.ft
        k = 1.4

        if rating == "LIDL":
            ti = [float(v.text) for v in self.badatree.findall("./PFM/TFM/LIDL/CT/ti")]
            ti_matrix = self.sci.reshape(ti, (3, 4))

            delta = self.aero.pressure(h) / self.aero.p0
            delta_powers = self.sci.array([delta**i for i in range(-1, 3)]).reshape(
                4, -1
            )
            mach_powers = self.sci.array([mach**i for i in range(3)]).reshape(3, -1)
            cT = self.sci.einsum("ij,jk,ik->k", ti_matrix, delta_powers, mach_powers)

        else:
            kink = float(self.badatree.findtext(f".//*/{rating}/kink"))

            if dT <= kink:
                bc_ = self.b_[rating]
            else:
                theta = self.aero.temperature(h) / self.aero.T0
                ratio = theta * (1 + (mach**2) * (k - 1) / 2)
                bc_ = self.c_[rating]

            a_matrix = self.sci.reshape(self.a_, (6, 6))
            bc_matrix = self.sci.reshape(bc_, (6, 6))

            mach_powers = self.sci.array([mach**i for i in range(6)]).reshape(6, -1)
            ratio_powers = self.sci.array([ratio**j for j in range(6)]).reshape(6, -1)

            delta_T = self.sci.einsum(
                "ij,jk,ik->k", bc_matrix, mach_powers, ratio_powers
            )
            delta_T_powers = self.sci.array([delta_T**j for j in range(6)]).reshape(
                6, -1
            )

            cT = self.sci.einsum("ij,jk,ik->k", a_matrix, mach_powers, delta_T_powers)

        return cT


# %%
class FuelFlow(fuel.FuelFlowBase):
    """Compute the fuel flow of an aircraft with BADA4 models."""

    def __init__(self, ac, bada_path, **kwargs):
        """Initialize FuelFlow object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            path (string): path to BADA4 models.

        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()
        self.thrust = Thrust(ac, bada_path)
        self.drag = Drag(ac, bada_path)

        # load parameters from xml
        bxml = load_bada4(ac, bada_path)
        self.mass_ref = float(bxml.findtext("./PFM/MREF"))
        self.fs = [float(v.text) for v in bxml.findall("./PFM/TFM/CF/f")]
        self.fis = [float(v.text) for v in bxml.findall("./PFM/TFM/LIDL/CF/fi")]
        self.lhv = float(bxml.findtext("./PFM/LHV"))

    @ndarrayconvert
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

    @ndarrayconvert
    def idle(self, mass, tas, alt, **kwargs):
        """Compute the fuel flow with idle rating.

        Args:
            mass (int or ndarray): Aircraft mass (unit: kg).
            tas (int or ndarray): Aircraft true airspeed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: Fuel flow (unit: kg/s).

        """

        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h)
        delta = self.aero.pressure(h) / self.aero.p0
        theta = self.aero.temperature(h) / self.aero.T0

        fi_matrix = self.sci.reshape(self.fis, (3, 3))
        delta_powers = self.sci.array([delta**i for i in range(3)]).reshape(3, -1)
        mach_powers = self.sci.array([mach**i for i in range(3)]).reshape(3, -1)

        cF_idle = self.sci.einsum("ij,jk,ik->k", fi_matrix, delta_powers, mach_powers)

        fuel_flow = self._calc_fuel(mass, delta, theta, cF_idle)

        return fuel_flow

    @ndarrayconvert
    def enroute(self, mass, tas, alt, vs=0, **kwargs):
        """Compute the fuel flow at not-idle.

        Args:
            mass (int or ndarray): Aircraft mass (unit: kg).
            tas (int or ndarray): Aircraft true airspeed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).
            vs (float or ndarray): Vertical rate (unit: ft/min). Default is 0.

        Returns:
            float: Fuel flow (unit: kg/s).

        """
        h = alt * self.aero.ft
        v = tas * self.aero.kts

        mach = self.aero.tas2mach(v, h)
        delta = self.aero.pressure(h) / self.aero.p0
        theta = self.aero.temperature(h) / self.aero.T0
        gamma = self.sci.arctan2(vs * self.aero.fpm, v)

        D = self.drag.clean(mass, tas, alt, vs)
        T = D + mass * self.aero.g0 * self.sci.sin(gamma)

        cT = T / (delta.reshape(-1, 1) * self.mass_ref * self.aero.g0)

        f_matrix = self.sci.reshape(self.fs, (5, 5))
        cT_powers = self.sci.array([cT[:, 0] ** i for i in range(5)]).reshape(5, -1)
        M_powers = self.sci.array([mach[:, 0] ** i for i in range(5)]).reshape(5, -1)

        cF_gen = self.sci.einsum("ij,jk,ik->k", f_matrix, cT_powers, M_powers)

        fuel_flow_non_idle = self._calc_fuel(mass, delta, theta, cF_gen)
        fuel_flow_idle = self.idle(mass, tas, alt)

        fuel_flow = self.sci.where(vs < -250, fuel_flow_idle, fuel_flow_non_idle)

        return fuel_flow
