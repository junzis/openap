# %%
from glob import glob
from xml.etree import ElementTree

from numpy import ndarray

from .. import base
from ..extra import ndarrayconvert


# %%
def load_bada4(ac: str, path: str) -> ElementTree.ElementTree:
    """Find and construct the drag polar model.
    Args:
        ac (str): aircraft type (for example: A320 or A320-231).
        bada_path (str): path to BADA4 models.

    Returns:
        xml.etree.ElementTree.ElementTree: BADA4 model XML.
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
    """
    Compute the drag of an aircraft using BADA4 models.

    Attributes:
        ac (str): aircraft type (e.g.: A320 or A320-231).
        scalar (float): Scalar value for drag coefficient calculation.
        d_ (ndarray): Array of drag coefficient parameters.
        mach_max (float): Maximum Mach number.
        S (float): Wing surface area (m^2).

    Methods:
        _cd_base(cl, mach):
            Compute the base drag coefficient (CD) for givenlift coefficient (CL)
            and Mach number.

        _cd(cl, mach):
            Compute the drag coefficient (CD) for given lift coefficient (CL)
            and Mach number, considering critical Mach number effects.

        _cl(mass, tas, alt, vs=0):
            Compute the lift coefficient (CL) for given aircraft mass,
            true airspeed (TAS), altitude, and vertical speed.

        clean(mass, tas, alt, vs=0):
            Compute the total drag (N) for the aircraft in clean configuration.
    """

    def __init__(self, ac, bada_path, **kwargs):
        """Initialize Drag object.

        Args:
            ac (str): aircraft type (for example: A320).
            path (str): path to BADA4 models.

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

    @ndarrayconvert(column=True)
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

    @ndarrayconvert(column=True)
    def _cd(self, cl, mach):
        """Compute the drag coefficient (CD)"""

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

    @ndarrayconvert(column=True)
    def _cl(self, mass, tas, alt, vs=0):
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        rho = self.aero.density(h)

        qS = 0.5 * rho * v**2 * self.S
        L = mass * self.aero.g0

        cl = L / self.sci.maximum(qS, 1e-3)  # avoid zero division

        return cl, qS

    @ndarrayconvert(column=True)
    def clean(self, mass, tas, alt, vs=0) -> float | ndarray:
        """Compute drag at clean configuration.

        Args:
            mass (float | ndarray): Mass of the aircraft (kg).
            tas (float | ndarray): True airspeed (kt).
            alt (float | ndarray): Altitude (ft).
            vs (float): Vertical rate (feet/min). Default: 0.

        Returns:
            float | ndarray: Total drag (N).

        """
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        mach = self.aero.tas2mach(v, h)

        cl, qS = self._cl(mass, tas, alt, vs)
        cd = self._cd(cl, mach)
        D = cd * qS

        return D


class Thrust(base.ThrustBase):
    """
    Thrust class for computing the thrust of an aircraft using BADA4 models.

    This class provides methods to compute the thrust coefficient and thrust force
    during different phases of flight such as climb, cruise, and takeoff.

    Attributes:
        ac (str): aircraft type (e.g.: A320 or A320-231).
        a_ (list): List of coefficients for thrust computation.
        b_ (dict): Dictionary of flat rating coefficients for different ratings.
        c_ (dict): Dictionary of temperature rating coefficients for different ratings.

    Methods:
        cT(mach, h, rating, dT=0):
            Compute the thrust coefficient considering altitude, Mach, and rating.
        climb(mass, tas, alt, dT=0):
            Compute the thrust force during climb phase.
        cruise(mass, tas, alt, dT=0):
            Compute the thrust force during cruise phase.
        takeoff(mass, tas, alt=0, dT=0):
            Compute the thrust force during takeoff phase.

    """

    def __init__(self, ac, bada_path, **kwargs):
        """Initialize Thrust object.

        Args:
            ac (str): aircraft type (for example: A320).
            path (str): path to BADA4 models.

        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()

        # load parameters from xml
        bxml = load_bada4(ac, bada_path)
        self.m_ref = float(bxml.findtext("./PFM/MREF"))
        self.a_ = [float(v.text) for v in bxml.findall("./PFM/TFM/CT/a")]
        self.ti = [float(v.text) for v in bxml.findall("./PFM/TFM/LIDL/CT/ti")]

        self.kink = dict()
        self.b_ = dict()
        self.c_ = dict()

        for rating in ["MCRZ", "MCMB"]:
            self.kink[rating] = float(bxml.findtext(f"./PFM/TFM/{rating}/kink"))
            self.b_[rating] = [
                float(t.text) for t in bxml.findall(f"./PFM/TFM/{rating}/flat_rating/b")
            ]

            self.c_[rating] = [
                float(t.text) for t in bxml.findall(f"./PFM/TFM/{rating}/temp_rating/c")
            ]

    @ndarrayconvert(column=True)
    def cT(self, mach, h, rating, dT=0) -> float | ndarray:
        """
        Compute the thrust coefficient, considering the altitude, Mach, and rating.
        """

        rating = rating.upper()
        assert rating in ["MCRZ", "MCMB", "LIDL"]

        k = 1.4

        delta = self.aero.pressure(h) / self.aero.p0
        theta = self.aero.temperature(h) / self.aero.T0

        if rating == "LIDL":
            ti_matrix = self.sci.reshape(self.ti, (3, 4))

            delta_pow = self.sci.array([delta**i for i in range(-1, 3)]).reshape(4, -1)
            mach_pow = self.sci.array([mach**i for i in range(3)]).reshape(3, -1)
            cT = self.sci.einsum("ij,jk,ik->k", ti_matrix, delta_pow, mach_pow)

        else:
            if dT <= self.kink[rating]:
                b_matrix = self.sci.reshape(self.b_[rating], (6, 6))
                mach_pow = self.sci.array([mach**i for i in range(6)]).reshape(6, -1)
                ratio_pow = self.sci.array([delta**j for j in range(6)]).reshape(6, -1)
                delta_T = self.sci.einsum("ij,jk,ik->k", b_matrix, mach_pow, ratio_pow)
            else:
                c_matrix = self.sci.reshape(self.c_[rating], (9, 5))
                mach_pow = self.sci.array([mach**i for i in range(5)]).reshape(5, -1)
                theta_t = theta * (1 + (mach**2) * (k - 1) / 2)
                ratio_pow = self.sci.array(
                    [theta_t**j for j in range(5)] + [delta**j for j in range(1, 5)]
                ).reshape(9, -1)
                delta_T = self.sci.einsum("ij,jk,ik->k", c_matrix, mach_pow, ratio_pow)

            a_matrix = self.sci.reshape(self.a_, (6, 6))
            mach_pow = self.sci.array([mach**i for i in range(6)]).reshape(6, -1)
            delta_T_pow = self.sci.array([delta_T**j for j in range(6)]).reshape(6, -1)
            cT = self.sci.einsum("ij,jk,ik->k", a_matrix, mach_pow, delta_T_pow)

        return cT

    @ndarrayconvert(column=True)
    def climb(self, tas, alt, dT=0) -> float | ndarray:
        """
        Compute the thrust force during the climb phase.

        Parameters:
            tas (float | ndarray): True airspeed (kts).
            alt (float | ndarray): Altitude (ft).
            dT (float | ndarray, optional): ISA temperature deviation (K). Default: 0.
        Returns:
            float | ndarray: The thrust force during the climb phase in Newtons.
        """
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h)
        delta = self.aero.pressure(h) / self.aero.p0

        cT = self.cT(mach, h, "MCMB", dT)

        return delta * self.m_ref * self.aero.g0 * cT

    @ndarrayconvert(column=True)
    def cruise(self, tas, alt, dT=0) -> float | ndarray:
        """
        Compute the thrust force during the cruise phase.

        Parameters:
            tas (float | ndarray): True airspeed (kts).
            alt (float | ndarray): Altitude (ft).
            dT (float | ndarray, optional): ISA temperature deviation (K). Default: 0.
        Returns:
            float | ndarray: The thrust force during the climb phase in Newtons.
        """
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h)
        delta = self.aero.pressure(h) / self.aero.p0

        cT = self.cT(mach, h, "MCRZ", dT)

        return delta * self.m_ref * self.aero.g0 * cT

    @ndarrayconvert(column=True)
    def takeoff(self, tas, alt=0, dT=0) -> float | ndarray:
        """
        Compute the thrust force at takeoff

        Parameters:
            tas (float | ndarray): True airspeed (kts).
            alt (float | ndarray): Altitude (ft). Default: 0.
            dT (float | ndarray, optional): ISA temperature deviation (K). Default: 0.
        Returns:
            float | ndarray: The thrust force during the climb phase in Newtons.
        """
        return self.climb(tas, alt=alt, dT=dT)

    @ndarrayconvert(column=True)
    def idle(self, tas, alt=0, dT=0) -> float | ndarray:
        """
        Compute the idle thrust

        Parameters:
            tas (float | ndarray): True airspeed (kts).
            alt (float | ndarray): Altitude (ft). Default: 0.
            dT (float | ndarray, optional): ISA temperature deviation (K). Default: 0.
        Returns:
            float | ndarray: The thrust force during the climb phase in Newtons.
        """
        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h)
        delta = self.aero.pressure(h) / self.aero.p0

        cT = self.cT(mach, h, "LIDL", dT)

        return delta * self.m_ref * self.aero.g0 * cT


# %%
class FuelFlow(base.FuelFlowBase):
    """
    FuelFlow class to compute the fuel flow of an aircraft using BADA4 models.

    Attributes:
        ac (str): aircraft type (e.g.: A320 or A320-231).
        mass_ref (float): Reference mass (kg).
        f_ (list): List of coefficients for fuel flow computation.
        fi_ (list): List of coefficients for idle rating.
        lhv (float): Lower heating value (J/kg).

    Methods:
        idle(mass, tas, alt, **kwargs):
            Compute the fuel flow with idle rating.

        enroute(mass, tas, alt, vs=0, **kwargs):
            Compute the fuel flow at not-idle.

    """

    def __init__(self, ac, bada_path, **kwargs):
        """Initialize FuelFlow object.

        Args:
            ac (str): aircraft type (for example: A320).
            path (str): path to BADA4 models.

        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()
        self.thrust = Thrust(ac, bada_path)
        self.drag = Drag(ac, bada_path)

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
            mass (float | ndarray): Aircraft mass (kg).
            tas (float | ndarray): Aircraft true airspeed (kt).
            alt (float | ndarray): Aircraft altitude (ft).

        Returns:
            float: Fuel flow (kg/s).

        """

        h = alt * self.aero.ft
        v = tas * self.aero.kts
        mach = self.aero.tas2mach(v, h)
        delta = self.aero.pressure(h) / self.aero.p0
        theta = self.aero.temperature(h) / self.aero.T0

        fi_matrix = self.sci.reshape(self.fi_, (3, 3))
        delta_powers = self.sci.array([delta**i for i in range(3)]).reshape(3, -1)
        mach_powers = self.sci.array([mach**i for i in range(3)]).reshape(3, -1)

        cF_idle = self.sci.einsum("ij,jk,ik->k", fi_matrix, delta_powers, mach_powers)

        fuel_flow = self._calc_fuel(mass, delta, theta, cF_idle)

        return fuel_flow

    @ndarrayconvert(column=True)
    def enroute(self, mass, tas, alt, vs=0, **kwargs) -> float | ndarray:
        """Compute the fuel flow at not-idle conditions.

        Args:
            mass (float | ndarray): Aircraft mass (kg).
            tas (float | ndarray): Aircraft true airspeed (kt).
            alt (float | ndarray): Aircraft altitude (ft).
            vs (float | ndarray): Vertical rate (ft/min). Default is 0.

        Returns:
            float: Fuel flow (kg/s).

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

        f_matrix = self.sci.reshape(self.f_, (5, 5))
        cT_powers = self.sci.array([cT[:, 0] ** i for i in range(5)]).reshape(5, -1)
        M_powers = self.sci.array([mach[:, 0] ** i for i in range(5)]).reshape(5, -1)

        cF_gen = self.sci.einsum("ij,jk,ik->k", f_matrix, cT_powers, M_powers)

        fuel_flow_non_idle = self._calc_fuel(mass, delta, theta, cF_gen)
        fuel_flow_idle = self.idle(mass, tas, alt)

        fuel_flow = self.sci.where(vs < -250, fuel_flow_idle, fuel_flow_non_idle)

        return fuel_flow
