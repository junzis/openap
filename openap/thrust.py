"""OpenAP thrust model.

Simplified two-shaft turbofan model based on:

- M. Bartel, T. M. Young, Simplified Thrust and Fuel Consumption
Models for Modern Two-Shaft Turbofan Engines

- C. Svoboda, Turbofan engine database as a preliminary design (cruise thrust)
"""

from typing import Optional

from openap import prop
from openap.backends import BackendType
from openap.extra import ndarrayconvert

from .base import ThrustBase


class Thrust(ThrustBase):
    """Simplified two-shaft turbofan model."""

    def __init__(
        self,
        ac: str,
        eng: Optional[str] = None,
        backend: Optional[BackendType] = None,
        **kwargs,
    ):
        """Initialize Thrust object.

        Args:
            ac: ICAO aircraft type (for example: A320).
            eng: Engine type (for example: CFM56-5A3).
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        super().__init__(ac, eng, backend=backend, **kwargs)

        aircraft = prop.aircraft(ac, **kwargs)
        force_engine = kwargs.get("force_engine", False)

        if eng is None:
            eng = aircraft["engine"]["default"]

        engine = prop.engine(eng.upper())

        eng_options = aircraft["engine"]["options"]

        if isinstance(eng_options, dict):
            eng_options = list(aircraft["engine"]["options"].values())

        if (not force_engine) and not any(
            opt.upper() in engine["name"].upper() for opt in eng_options
        ):
            raise ValueError(
                (
                    f"Engine {eng} and aircraft {ac} mismatch. "
                    f"Available engines for {ac} are {eng_options}"
                )
            )

        self.cruise_alt = aircraft["cruise"]["height"] / self.aero.ft
        self.eng_bpr = engine["bpr"]
        self.eng_max_thrust = engine["max_thrust"]
        self.eng_number = aircraft["engine"]["number"]

        if engine["cruise_mach"] > 0:
            self.cruise_mach = engine["cruise_mach"]
            self.eng_cruise_thrust = engine["cruise_thrust"]
        else:
            self.cruise_mach = aircraft["cruise"]["mach"]
            self.eng_cruise_thrust = 0.2 * self.eng_max_thrust + 890

    def _dfunc(self, mratio):
        """Compute parameter 'd' from Equation 15 in Bartel and Young (2008).

        Linear fit to the data from Table 2.

        Args:
            mratio: Ratio of mach number to reference mach number.

        Returns:
            Parameter 'd'.
        """
        d = -0.4204 * mratio + 1.0824
        return d

    def _nfunc(self, roc):
        """Compute parameter 'n' from Table 3 in Bartel and Young (2008).

        Linear fit assuming climb rates:
        - Fast climb: 4000 ft/min
        - Moderate climb: 2500 ft/min
        - Slow climb: 1000 ft/min

        Args:
            roc: Rate of climb (ft/min).

        Returns:
            Parameter 'n'.
        """
        n = 2.667e-05 * roc + 0.8633
        return n

    def _mfunc(self, vratio, roc):
        """Compute parameter 'm' from Table 4 in Bartel and Young (2008).

        Args:
            vratio: Ratio of CAS to reference CAS.
            roc: Rate of climb (ft/min).

        Returns:
            Parameter 'm'.
        """
        m = -1.2043e-1 * vratio - 8.8889e-9 * roc**2 + 2.4444e-5 * roc + 4.7379e-1
        return m

    @ndarrayconvert
    def takeoff(self, tas, alt=0, dT=0):
        """Calculate thrust at take-off condition.

        Args:
            tas: True airspeed (kt).
            alt: Altitude of the runway (ft). Defaults to 0.
            dT: Temperature shift (K or degC). Defaults to 0.

        Returns:
            Total thrust (N).
        """
        b = self.backend

        # Flight mach number
        mach = self.aero.tas2mach(tas * self.aero.kts, 0, dT=dT)

        # Engine bypass ratio
        eng_bpr = self.eng_bpr

        # G0 is the "Gas generator function", defined as the ratio of
        # the kinetic energy to the flow of enthalpy into the jet core
        # This is a fit to Fig. 5 in Bartel and Young (2008)
        G0 = 0.0606 * self.eng_bpr + 0.6337

        P = self.aero.pressure(alt * self.aero.ft, dT=dT)
        dP = P / self.aero.p0

        # Equations 12, 13 and 14 in Bartel and Young (2008)
        # Evaluate to 1 if dP = 1, which is the case for alt = 0 ft
        A = -0.4327 * dP**2 + 1.3855 * dP + 0.0472
        Z = 0.9106 * dP**3 - 1.7736 * dP**2 + 1.8697 * dP
        X = 0.1377 * dP**3 - 0.4374 * dP**2 + 1.3003 * dP

        # Equation 11 in Bartel and Young (2008)
        ratio = (
            A
            - 0.377 * (1 + eng_bpr) / b.sqrt((1 + 0.82 * eng_bpr) * G0) * Z * mach
            + (0.23 + 0.19 * b.sqrt(eng_bpr)) * X * mach**2
        )

        F = ratio * self.eng_max_thrust * self.eng_number
        return F

    @ndarrayconvert
    def cruise(self, tas, alt, dT=0):
        """Calculate thrust at cruise.

        Args:
            tas: True airspeed (kt).
            alt: Altitude (ft).
            dT: Temperature shift (K or degC). Defaults to 0.

        Returns:
            Total thrust (N).
        """
        return self.climb(tas, alt, roc=0, dT=dT)

    @ndarrayconvert
    def climb(self, tas, alt, roc, dT=0):
        """Calculate thrust during climb.

        Args:
            tas: True airspeed (kt).
            alt: Altitude (ft).
            roc: Vertical rate (ft/min).
            dT: Temperature shift (K or degC). Defaults to 0.

        Returns:
            Total thrust (N).
        """
        b = self.backend

        roc = b.abs(roc)

        h = alt * self.aero.ft
        tas = b.maximum(10, tas)

        mach = self.aero.tas2mach(tas * self.aero.kts, h, dT=dT)
        vcas = self.aero.tas2cas(tas * self.aero.kts, h, dT=dT)

        P = self.aero.pressure(h, dT=dT)

        P10 = self.aero.pressure(10000 * self.aero.ft, dT=dT)
        Pcr = self.aero.pressure(self.cruise_alt * self.aero.ft, dT=dT)

        # approximate thrust at top of climb
        Fcr = self.eng_cruise_thrust * self.eng_number
        vcas_ref = self.aero.mach2cas(
            self.cruise_mach, self.cruise_alt * self.aero.ft, dT=dT
        )

        # segment 3: alt > 30000:
        d = self._dfunc(mach / self.cruise_mach)

        # Equation 16 in Bartel and Young (2008)
        bcoef = (mach / self.cruise_mach) ** (-0.11)

        # Equation 15 in Bartel and Young (2008)
        ratio_seg3 = d * b.log(P / Pcr) + bcoef

        # segment 2: 10000 < alt <= 30000:
        # Equation 18 in Bartel and Young (2008)
        a = (vcas / vcas_ref) ** (-0.1)
        n = self._nfunc(roc)

        # Equation 17 in Bartel and Young (2008)
        ratio_seg2 = a * (P / Pcr) ** (-0.355 * (vcas / vcas_ref) + n)

        # segment 1: alt <= 10000:
        # Equation 17 in Bartel and Young (2008)
        F10 = Fcr * a * (P10 / Pcr) ** (-0.355 * (vcas / vcas_ref) + n)
        m = self._mfunc(vcas / vcas_ref, roc)

        # Equation 19 in Bartel and Young (2008)
        ratio_seg1 = m * (P / Pcr) + (F10 / Fcr - m * (P10 / Pcr))

        ratio = b.where(
            alt > 30000, ratio_seg3, b.where(alt > 10000, ratio_seg2, ratio_seg1)
        )

        F = ratio * Fcr
        return F

    def descent_idle(self, tas, alt, dT=0):
        """Calculate idle thrust during descent.

        Note: The idle thrust at the descent is taken as 7% of the maximum
        available thrust. This is an approximation.

        Args:
            tas: True airspeed (kt).
            alt: Altitude (ft).
            dT: Temperature shift (K or degC). Defaults to 0.

        Returns:
            Total thrust (N).
        """
        F = 0.07 * self.takeoff(tas, alt, dT=dT)
        return F
