"""OpenAP FuelFlow model."""

import numpy as np
from openap.extra import aero
from openap import prop, FuelFlow
from openap.extra import ndarrayconvert


class Emission(object):
    """Emission model based on ICAO emmision databank."""

    def __init__(self, ac, eng=None):
        """Initialize FuelFlow object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            eng (string): Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                by in the aircraft database.

        """
        self.ac = prop.aircraft(ac)
        self.n_eng = self.ac["engine"]["number"]

        if eng is None:
            eng = self.ac["engine"]["default"]

        self.engine = prop.engine(eng)

    def _sl2fl(self, tas, alt):
        M = aero.tas2mach(tas * aero.kts, alt * aero.ft)
        beta = np.exp(0.2 * (M ** 2))
        theta = (aero.temperature(alt * aero.ft) / 288.15) / beta
        delta = (1 - 0.0019812 * alt / 288.15) ** 5.255876 / np.power(beta, 3.5)
        ratio = (theta ** 3.3) / (delta ** 1.02)
        return beta, theta, delta, ratio

    # @ndarrayconvert
    def co2(self, ffac):
        """Compute CO2 emission at given fuel flow.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).

        Returns:
            float: CO2 emission from all engines (unit: g/s).

        """
        return ffac * 3149

    # @ndarrayconvert
    def h2o(self, ffac):
        """Compute H2O emission at given fuel flow.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).

        Returns:
            float: H2O emission from all engines (unit: g/s).

        """
        return ffac * 1230

    # @ndarrayconvert
    def nox(self, ffac, tas, alt=0):
        """Compute NOx emission at given fuel flow, speed, and altitude.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).
            tas (float or ndarray): Speed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: NOx emission from all engines (unit: g/s).

        """
        beta, theta, delta, ratio = self._sl2fl(tas, alt)

        ffsl = (ffac / self.n_eng) * theta ** 3.8 / delta * beta

        nox_sl = self.engine["nox_c"] * (ffsl ** self.engine["nox_p"])

        # convert to actual flight level
        omega = 10 ** (-3) * np.exp(-0.0001426 * (alt - 12900))
        nox_fl = nox_sl * np.sqrt(1 / ratio) * np.exp(-19 * (omega - 0.00634))

        # convert g/(kg fuel) to g/s for all engines
        nox_rate = nox_fl * ffac
        return nox_rate

    # @ndarrayconvert
    def co(self, ffac, tas, alt=0):
        """Compute CO emission at given fuel flow, speed, and altitude.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).
            tas (float or ndarray): Speed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: CO emission from all engines (unit: g/s).

        """
        beta, theta, delta, ratio = self._sl2fl(tas, alt)

        ffsl = (ffac / self.n_eng) * theta ** 3.8 / delta * beta

        co_beta = self.engine["co_beta"]
        co_gamma = self.engine["co_gamma"]
        co_min = self.engine["co_min"]
        co_max = self.engine["co_max"]

        ffsl = np.maximum(ffsl, 0.001)
        co_sl = (
            co_beta
            * (ffsl - 0.001) ** (-co_gamma)
            * np.exp(-2 * (ffsl - 0.001) ** co_beta)
        )

        co_sl = np.where(co_sl < co_min, co_min, co_sl)
        co_sl = np.where(co_sl > co_max, co_max, co_sl)

        # convert to actual flight level
        co_fl = co_sl * ratio

        # convert g/(kg fuel) to g/s for all engines
        co_rate = co_fl * ffac
        return co_rate

    # @ndarrayconvert
    def hc(self, ffac, tas, alt=0):
        """Compute HC emission at given fuel flow, speed, and altitude.

        Args:
            ffac (float or ndarray): Fuel flow for all engines (unit: kg/s).
            tas (float or ndarray): Speed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).

        Returns:
            float: HC emission from all engines (unit: g/s).

        """
        beta, theta, delta, ratio = self._sl2fl(tas, alt)

        ffsl = (ffac / self.n_eng) * theta ** 3.8 / delta * beta

        hc_beta = self.engine["hc_beta"]
        hc_gamma = self.engine["hc_gamma"]
        a1 = self.engine["hc_a1"]
        b1 = self.engine["hc_b1"]
        b2 = self.engine["hc_b2"]
        ff85 = self.engine["hc_ff85"]
        hc_na = self.engine["hc_na"]
        hc_max = self.engine["hc_max"]
        hc_min = self.engine["hc_min"]

        if hc_na:
            return None

        if hc_beta is not None:
            ffsl = np.maximum(ffsl, 0.001)
            hc_sl = (
                hc_beta
                * (ffsl + 0.05) ** (-hc_gamma)
                * np.exp(-4 * (ffsl - 0.001) ** hc_beta)
            )
        else:
            hc_sl = 10 ** (a1 * np.log10(ffsl) + b1)
            if ff85 > 0:
                hc_sl = np.where(ffsl > ff85, 10 ** b2, hc_sl)

        hc_sl = np.where(hc_sl > hc_max, hc_max, hc_sl)
        hc_sl = np.where(hc_sl < hc_min, hc_min, hc_sl)

        # convert to actual flight level
        hc_fl = hc_sl * ratio

        # convert g/(kg fuel) to g/s for all engines
        hc_rate = hc_fl * ffac
        return hc_rate
