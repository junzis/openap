"""OpenAP FuelFlow model."""

import importlib

from openap import prop
from openap.extra import ndarrayconvert


class Emission(object):
    """Emission model based on ICAO emmision databank."""

    def __init__(self, ac, eng=None, **kwargs):
        """Initialize Emission object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            eng (string): Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                by in the aircraft database.

        """
        if not hasattr(self, "np"):
            self.sci = importlib.import_module("numpy")

        if not hasattr(self, "aero"):
            self.aero = importlib.import_module("openap").aero

        self.ac = prop.aircraft(ac, **kwargs)
        self.n_eng = self.ac["engine"]["number"]

        if eng is None:
            eng = self.ac["engine"]["default"]

        self.engine = prop.engine(eng)

    def _fl2sl(self, ffac, tas, alt):
        """Convert to sea-level equivalent"""
        M = self.aero.tas2mach(tas * self.aero.kts, alt * self.aero.ft)
        beta = self.sci.exp(0.2 * (M**2))
        theta = (self.aero.temperature(alt * self.aero.ft) / 288.15) / beta
        delta = (1 - 0.0019812 * alt / 288.15) ** 5.255876 / self.sci.power(beta, 3.5)
        ratio = (theta**3.3) / (delta**1.02)
        # TODO: Where does this equation come from?
        ff_sl = (ffac / self.n_eng) * theta**3.8 / delta * beta

        return ff_sl, ratio

    @ndarrayconvert
    def co2(self, ffac):
        """Compute CO2 emission with given fuel flow.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).

        Returns:
            float: CO2 emission from all engines (unit: g/s).

        """
        # IATA: jet fuel -> co2
        return ffac * 3160

    @ndarrayconvert
    def h2o(self, ffac):
        """Compute H2O emission with given fuel flow.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).

        Returns:
            float: H2O emission from all engines (unit: g/s).

        """
        # kerosene -> water
        return ffac * 1230

    @ndarrayconvert
    def soot(self, ffac):
        """Compute soot emission with given fuel flow.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).

        Returns:
            float: Soot emission from all engines (unit: g/s).

        """
        # Barrett et al. 2010 - Global Mortality Attributable to Aircraft Cruise Emissions
        return ffac * 0.03

    @ndarrayconvert
    def sox(self, ffac):
        """Compute SOx emission with given fuel flow.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).

        Returns:
            float: SOx emission from all engines (unit: g/s).

        """
        # Barrett et al. 2010 - Global Mortality Attributable to Aircraft Cruise Emissions
        return ffac * 1.2

    @ndarrayconvert
    def nox(self, ffac, tas, alt=0):
        """Compute NOx emission with given fuel flow, speed, and altitude.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).
            tas (float or ndarray): Speed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: NOx emission from all engines (unit: g/s).

        """
        ff_sl, ratio = self._fl2sl(ffac, tas, alt)

        nox_sl = self.sci.interp(
            ff_sl,
            [
                self.engine["ff_idl"],
                self.engine["ff_app"],
                self.engine["ff_co"],
                self.engine["ff_to"],
            ],
            [
                self.engine["ei_nox_idl"],
                self.engine["ei_nox_app"],
                self.engine["ei_nox_co"],
                self.engine["ei_nox_to"],
            ],
        )

        # convert back to actual flight level
        omega = 10 ** (-3) * self.sci.exp(-0.0001426 * (alt - 12900))

        # TODO: source
        nox_fl = (
            nox_sl * self.sci.sqrt(1 / ratio) * self.sci.exp(-19 * (omega - 0.00634))
        )

        # convert g/(kg fuel) to g/s for all engines
        nox_rate = nox_fl * ffac
        return nox_rate

    @ndarrayconvert
    def co(self, ffac, tas, alt=0):
        """Compute CO emission with given fuel flow, speed, and altitude.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).
            tas (float or ndarray): Speed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: CO emission from all engines (unit: g/s).

        """
        ff_sl, ratio = self._fl2sl(ffac, tas, alt)

        co_sl = self.sci.interp(
            ff_sl,
            [
                self.engine["ff_idl"],
                self.engine["ff_app"],
                self.engine["ff_co"],
                self.engine["ff_to"],
            ],
            [
                self.engine["ei_co_idl"],
                self.engine["ei_co_app"],
                self.engine["ei_co_co"],
                self.engine["ei_co_to"],
            ],
        )

        # TODO: source
        # convert back to actual flight level
        co_fl = co_sl * ratio

        # convert g/(kg fuel) to g/s for all engines
        co_rate = co_fl * ffac
        return co_rate

    @ndarrayconvert
    def hc(self, ffac, tas, alt=0):
        """Compute HC emission with given fuel flow, speed, and altitude.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).
            tas (float or ndarray): Speed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: HC emission from all engines (unit: g/s).

        """
        ff_sl, ratio = self._fl2sl(ffac, tas, alt)

        hc_sl = self.sci.interp(
            ff_sl,
            [
                self.engine["ff_idl"],
                self.engine["ff_app"],
                self.engine["ff_co"],
                self.engine["ff_to"],
            ],
            [
                self.engine["ei_hc_idl"],
                self.engine["ei_hc_app"],
                self.engine["ei_hc_co"],
                self.engine["ei_hc_to"],
            ],
        )
        # TODO: source
        # convert back to actual flight level
        hc_fl = hc_sl * ratio

        # convert g/(kg fuel) to g/s for all engines
        hc_rate = hc_fl * ffac
        return hc_rate
