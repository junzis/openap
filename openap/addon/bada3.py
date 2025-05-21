"""
Implements BADA3.
"""

from numpy import ndarray
from pyBADA.bada3 import Bada3Aircraft

from .. import base
from ..extra import ndarrayconvert

def load_bada3(ac:str, bada_version:str, bada_path:str=None):
    """
    Load BADA3 aircraft.

    Args:
        ac (str): aircraft ICAO identifier (e.g. A320)
        bada_version (str): identifier of BADA version. Required if
            bada_path=None,  else has no functionality but must be given.
        bada_path (str, optional): path to BADA3 models. If None, data is taken
            from `pyData/aircraft/BADA3/{badaVersion}/`.
    """

    model = Bada3Aircraft(
        badaVersion=bada_version, acName=ac, filePath=bada_path
    )

    return model


class Drag(base.DragBase):
    """
    Compute the drag of an aircraft using BADA3 models.
    
    Attributes:
        ac (str): aircraft ICAO identifer (e.g. A320)
        cd0_cr (float): parasitic drag coefficient (cruise)
        cd2_cr (float): induced drag coefficient (cruise)
        S (float): wing surface area (m^2)

    Methods:
        _cd(cl):
            Compute the drag coefficient for a given lift coefficient.

        _cl(mass, tas, alt, vs=0):
            Compute the lift coefficient for a given aircraft mass, true
            airspeed (TAS), altitude and vertical speed.

        clean(mass, tas, alt, vs=0):
            Compute the total drag (N) for the aircraft in clean configuration.
    """

    def __init__(self, ac:str, bada_version:str, bada_path:str=None, **kwargs):
        """
        Initialise Drag object.
        
        Args:
            ac (str): aircraft ICAO identifier (e.g. A320)
            bada_version (str): identifier of BADA3 version. Required if
                bada_path=None, else has no functionality but must be given.
            bada_path (str, optional): path to BADA3 models. If None, data is
                taken from `pyData/aircraft/BADA3/{badaVersion}/`.
        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()

        # load parameters from BADA3
        model = load_bada3(ac, bada_version, bada_path)
        self.cd0_cr = model.CD0["CR"]
        self.cd2_cr = model.CD2["CR"]
        self.S = model.S

    @ndarrayconvert(column=True)
    def _cd(self, cl):
        """Compute drag coefficient."""

        cd = self.cd0_cr + self.cd2_cr * cl ** 2
        return cd

    @ndarrayconvert(column=True)
    def _cl(self, mass, tas, alt, vs=0):
        """Compute lift coefficient."""

        v = tas * self.aero.kts
        h = alt * self.aero.ft
        rho = self.aero.density(h)

        qS = 0.5 * rho * v**2 * self.S
        L = mass * self.aero.g0

        cl = L / self.sci.maximum(qS, 1e-3)  # avoid zero division

        return cl, qS

    @ndarrayconvert(column=True)
    def clean(self, mass, tas, alt, vs=0) -> float | ndarray:
        """
        Compute drag at clean configuration.

        Args:
            mass (float | ndarray): Mass of the aircraft (kg).
            tas (float | ndarray): True airspeed (kt).
            alt (float | ndarray): Altitude (ft).
            vs (float): Vertical rate (feet/min). Default: 0.

        Returns:
            float | ndarray: Total drag (N).

        """
        cl, qS = self._cl(mass, tas, alt, vs)
        cd = self._cd(cl)
        D = cd * qS

        return D


class Thrust(base.ThrustBase):
    """
    Thrust class for computing the thrust of an aircraft using BADA3 models.
    
    This class provides methods to compute thrust during the flight phases
    takeoff, climb and cruise.
    
    Attributes:
        ac (str): aircraft ICAO identifer (e.g. A320)
        engine_type (str): one of JET, TURBOPROP, PISTON or ELECTRIC
        ct (list): list of thrust-related coefficients from BADA3
        hpdes (float): design geopotential pressure altitude [ft]

    Methods:
        climb(tas, alt, dT):
            Compute the thrust force during the climb phase.

        cruise(tas, alt, dT):
            Compute the thrust force during the cruise phase.

        takeoff(tas, alt, dT):
            Compute the thrust force during the takeoff phase.
    """
    
    def __init__(self, ac:str, bada_version:str, bada_path:str=None, **kwargs):
        """Initialise Thrust object.
        
        Args:
            ac (str): aircraft ICAO identifier (e.g. A320)
            bada_version (str): identifier of BADA3 version. Required if
                bada_path=None, else has no functionality but must be given.
            bada_path (str, optional): path to BADA3 models. If None, data is
                taken from `pyData/aircraft/BADA3/{badaVersion}/`.
        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()

        # load parameters from BADA3
        model = load_bada3(ac, bada_version, bada_path)
        self.engine_type = model.engineType
        self.hpdes = model.HpDes
        self.ct = model.Ct


    @ndarrayconvert(column=True)
    def climb(self, tas, alt, dT=0) -> float | ndarray:
        """
        Compute the maximum climb thrust.

        Args:
            tas (float | ndarray): true airspeed [kts]
            alt (float | ndarray): geopotential pressure altitude [ft]
            dT (float | ndarray, optional): ISA temperature deviation [K]. 
                Defaults to 0.0 K.

        Returns:
            float | ndarray: Thrust force (N) during the climb phase.
        """
        
        # calculate maximum climb thrust at ISA
        if self.engine_type == "JET":
            thr_isa = self.ct[0] * (1 - alt / self.ct[1] + self.ct[2] * alt ** 2)
        elif self.engine_type == "TURBOPROP":
            thr_isa = self.ct[0] / tas * (1 - alt / self.ct[1]) + self.ct[2]
        elif self.engine_type in ("PISTON", "ELECTRIC"):
            thr_isa = self.ct[0] * (1 - alt / self.ct[1]) + self.ct[2] / tas
        else:
            raise ValueError("Unknown engine type")
        
        # correct for temperature deviations from ISA
        dT_eff = dT - self.ct[3]
        thr_mcl = thr_isa * (1 - self.ct[4] * dT_eff)
        return thr_mcl

    # @ndarrayconvert(column=True)
    # def reduced_climb(self):
        
    #     return

    @ndarrayconvert(column=True)
    def cruise(self, tas, alt, dT=0) -> float | ndarray:
        """
        Compute the maximum cruise thrust.

        Args:
            tas (float | ndarray): true airspeed [kts]
            alt (float | ndarray): geopotential pressure altitude [ft]
            dT (float | ndarray, optional): ISA temperature deviation [K]. 
                Defaults to 0.0 K.

        Returns:
            float | ndarray: Thrust force (N) during the cruise phase.
        """
        ct_cr = 0.95  # constant in BADA3
        thr_cl_max = self.climb(tas, alt, dT)
        thr_cr_max = ct_cr * thr_cl_max
        return thr_cr_max

    @ndarrayconvert(column=True)
    def takeoff(self, tas, alt, dT=0) -> float | ndarray:
        """
        Compute takeoff thrust, which is assumed by BADA3 to be equal to 
        maximum climb thrust.

        Args:
            tas (float | ndarray): true airspeed [kts]
            alt (float | ndarray): geopotential pressure altitude [ft]
            dT (float | ndarray, optional): ISA temperature deviation [K]. 
                Defaults to 0.0 K.

        Returns:
            float | ndarray: Thrust force (N) during the cruise phase.
        """
        return self.climb(tas, alt, dT)


class FuelFlow(base.FuelFlowBase):
    """
    FuelFlow class to compute the fuel flow of an aircraft using BADA3 models.
    
    Attributes:
        ac (str): aircraft ICAO identifier (e.g. A320)
        engine_type (str): one of JET, TURBOPROP, PISTON or ELECTRIC
        cf1 (float): 1st thrust specific fuel consumption coefficient
        cf2 (float): 2nd thrust specific fuel consumption coefficient
        cf3 (float): 1st descent fuel flow coefficient
        cf4 (float): 2nd descent fuel flow coefficient
        cfcr (float): cruise fuel flow correction coefficient

    Args:
        base (_type_): _description_
    """

    def __init__(self, ac:str, bada_version:str, bada_path:str=None, **kwargs):
        """Initialise FuelFlow object.
        
        Args:
            ac (str): aircraft ICAO identifier (e.g. A320)
            bada_version (str): identifier of BADA3 version. Required if
                bada_path=None, else has no functionality but must be given.
            bada_path (str, optional): path to BADA3 models. If None, data is
                taken from `pyData/aircraft/BADA3/{badaVersion}/`.
        """
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()
        self.thrust = Thrust(ac, bada_version, bada_path)
        self.drag = Drag(ac, bada_version, bada_path)

        # load parameters from BADA3
        model = load_bada3(ac, bada_version, bada_path)
        self.engine_type = model.engineType
        self.cf1 = model.Cf[0]
        self.cf2 = model.Cf[1]
        self.cf3 = model.CfDes[0]
        self.cf4 = model.CfDes[1]
        self.cfcr = model.CfCrz

    @ndarrayconvert(column=True)
    def nominal(self, mass, tas, alt, vs=0):
        """Calculate the nominal fuel flow.

        Args:
            mass (float | ndarray): Mass of the aircraft (kg).
            tas (float | ndarray): True airspeed (kt).
            alt (float | ndarray): Altitude (ft).
            vs (float): Vertical rate (feet/min). Default: 0.

        Returns:
            float | ndarray: Nominal fuel flow [kg/min]
        """
        v = tas * self.aero.kts
        gamma = self.sci.arctan2(vs * self.aero.fpm, v)
        D = self.drag.clean(mass, tas, alt, vs)
        T = D + mass * self.aero.g0 * self.sci.sin(gamma)

        if self.engine_type == "JET":
            eta = self.cf1 * (1 + tas / self.cf2) * 1e-3
            f_nom = eta * T
        elif self.engine_type == "TURBOPROP":
            eta = self.cf1 * (1 - tas / self.cf2) * (tas / 1e3) * 1e-3
            f_nom = eta * T
        elif self.engine_type in ("PISTON", "ELECTRIC"):
            f_nom = self.cf1
        else:
            raise ValueError("Unknown engine type.")

        return f_nom

    @ndarrayconvert(column=True)
    def cruise(self, mass, tas, alt, vs=0):
        """Calculate the cruise fuel flow.

        Args:
            mass (float | ndarray): Mass of the aircraft (kg).
            tas (float | ndarray): True airspeed (kt).
            alt (float | ndarray): Altitude (ft).
            vs (float): Vertical rate (feet/min). Default: 0.

        Returns:
            float | ndarray: Cruise fuel flow [kg/min]
        """
        f_nom = self.nominal(mass, tas, alt, vs)
        return self.cfcr * f_nom

    @ndarrayconvert(column=True)
    def idle(self, alt):
        """Calculate idle fuel flow.

        Args:
            alt (float | ndarray): Geopotential pressure altitude [ft]

        Returns:
            float | ndarray: Idle fuel flow [kg/min]
        """
        if self.engine_type in ("JET", "TURBOPROP"):
            f_min = self.cf3 * (1 - alt / self.cf4)
        elif self.engine_type in ("PISTON", "ELECTRIC"):
            f_min = self.cf3
        else:
            raise ValueError("Unknown engine type.")
        return f_min
