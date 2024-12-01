"""Trajectory generator based on WRAP model.

This module defines the Generator class than can be used to generate flight
trajectories using the WRAP kinematic performance model from OpenAP.

Examples::

    trajgen = Generator(ac='a320')

    trajgen.enable_noise()   # enable Gaussian noise in trajectory data

    df_cl = trajgen.climb(dt=10, cas_const_cl=280, mach_const_cl=0.78, alt_cr=35000)
    df_cl = trajgen.climb(dt=10, random=True)  # using radomparameters

    df_de = trajgen.descent(dt=10, cas_const_de=280, mach_const_de=0.78, alt_cr=35000)
    df_de = trajgen.descent(dt=10, random=True)

    df_cr = trajgen.cruise(dt=60, range_cr=2000, alt_cr=35000, m_cr=0.78)
    df_cr = trajgen.cruise(dt=60, random=True)

    df_all = trajgen.complete(dt=10, alt_cr=35000, m_cr=0.78,
                                cas_const_cl=280, mach_const_cl=0.78,
                                cas_const_de=280, mach_const_de=0.78)
    df_all = trajgen.complete(dt=10, random=True)

"""

import numpy as np
import pandas as pd
from openap import aero, prop

from .kinematic import WRAP


class FlightGenerator:
    """Generate trajectory using WRAP model."""

    def __init__(self, ac: str, random_seed: int = 42) -> None:
        """Intitialize the generator.

        Args:
            ac (str): ICAO aircraft type (for example: A320).
            random_seed (int): Random seed for noise generation.

        """
        super(FlightGenerator, self).__init__()

        self.ac = ac
        self.acdict = prop.aircraft(self.ac)

        self.wrap = WRAP(self.ac)

        # for noise generation
        self.sigma_v = 0
        self.sigma_vs = 0
        self.sigma_h = 0
        self.sigma_s = 0

        self.rng = np.random.default_rng(random_seed)
        self.noise = False

    def enable_noise(self) -> None:
        """Adding noise to the generated trajectory.

        The noise model is based on ADS-B Version 1&2, NACv=3 and NACp=10
        """
        self.sigma_v = 0.5
        self.sigma_vs = 0.75
        self.sigma_h = 7.5
        self.sigma_s = 5
        self.noise = True

    def climb(self, **kwargs) -> pd.DataFrame:
        """Generate climb trajectory based on WRAP model.

        Args:
            **dt (int): Time step in seconds.
            **cas_const_cl (int): Constant CAS for climb (kt).
            **mach_const_cl (float): Constant Mach for climb (-).
            **alt_cr (int): Target cruise altitude (ft).
            **random (bool): Generate trajectory with random parameters.

        """
        dt = kwargs.get("dt", 1)
        random = kwargs.get("random", False)

        a_tof = self.wrap.takeoff_acceleration()["default"]
        v_tof = self.wrap.takeoff_speed()["default"]

        if random:
            cas_const = kwargs.get(
                "cas_const_cl",
                self.rng.uniform(
                    self.wrap.climb_const_vcas()["minimum"],
                    self.wrap.climb_const_vcas()["maximum"],
                )
                / aero.kts,
            )

            mach_const = kwargs.get(
                "mach_const_cl",
                self.rng.uniform(
                    self.wrap.climb_const_mach()["minimum"],
                    self.wrap.climb_const_mach()["maximum"],
                ),
            )

            alt_cr = kwargs.get(
                "alt_cr",
                self.rng.uniform(
                    self.wrap.cruise_alt()["minimum"], self.wrap.cruise_alt()["maximum"]
                )
                * 1000
                / aero.ft,
            )

            vs_pre_constcas = self.rng.uniform(
                self.wrap.climb_vs_pre_concas()["minimum"],
                self.wrap.climb_vs_pre_concas()["maximum"],
            )

            vs_constcas = self.rng.uniform(
                self.wrap.climb_vs_concas()["minimum"],
                self.wrap.climb_vs_concas()["maximum"],
            )

            vs_constmach = self.rng.uniform(
                self.wrap.climb_vs_conmach()["minimum"],
                self.wrap.climb_vs_conmach()["maximum"],
            )

        else:
            cas_const = kwargs.get(
                "cas_const_cl", self.wrap.climb_const_vcas()["default"] / aero.kts
            )

            # cas can not be smaller then takeoff speed
            cas_const = max(cas_const, v_tof / aero.kts)
            mach_const = kwargs.get(
                "mach_const_cl", self.wrap.climb_const_mach()["default"]
            )
            alt_cr = kwargs.get(
                "alt_cr", self.wrap.cruise_alt()["default"] * 1000 / aero.ft
            )
            vs_pre_constcas = self.wrap.climb_vs_pre_concas()["default"]
            vs_constcas = self.wrap.climb_vs_concas()["default"]
            vs_constmach = self.wrap.climb_vs_conmach()["default"]

        vcas_const = cas_const * aero.kts
        alt_cr = np.round(alt_cr, -2)  # round to flight level
        h_cr = alt_cr * aero.ft
        vs_ic = self.wrap.initclimb_vs()["default"]
        h_const_cas = self.wrap.climb_cross_alt_concas()["default"] * 1000

        h_const_mach = aero.crossover_alt(vcas_const, mach_const)

        data = []

        # intitial conditions
        t = 0
        tcr = 0
        h = 0
        s = 0
        v = 0
        vs = 0
        a = 0.5  # standard acceleration m/s^2
        seg = None

        while True:
            data.append([t, h, s, v, vs, seg])
            t = t + dt
            s = s + v * dt
            h = h + vs * dt

            if v < v_tof:
                v = v + a_tof * dt
                vs = 0
                seg = "TO"
            elif h < 1500 * aero.ft:
                v = v + a * dt
                if aero.tas2cas(v, h) >= vcas_const:
                    v = aero.cas2tas(vcas_const, h)
                vs = vs_ic
                seg = "IC"
            elif h < h_const_cas:
                v = v + a * dt
                if aero.tas2cas(v, h) >= vcas_const:
                    v = aero.cas2tas(vcas_const, h)
                vs = vs_pre_constcas
                seg = "PRE-CAS"
            elif h < h_const_mach:
                v = aero.cas2tas(vcas_const, h)
                vs = vs_constcas
                seg = "CAS"
            elif h < h_cr:
                v = aero.mach2tas(mach_const, h)
                vs = vs_constmach
                seg = "MACH"
            else:
                v = aero.mach2tas(mach_const, h)
                vs = 0
                seg = "CR"
                if tcr == 0:
                    tcr = t
                if t - tcr > 60:
                    break

        df = pd.DataFrame(data, columns=["t", "h", "s", "v", "vs", "seg"])

        if self.noise:
            n = df.shape[0]
            df = df.eval("""
            h = h + @self.rng.normal(0, @self.sigma_h, @n)
            s = s + @self.rng.normal(0, @self.sigma_s, @n)
            v = v + @self.rng.normal(0, @self.sigma_v, @n)
            vs = vs + @self.rng.normal(0, @self.sigma_vs, @n)
            """)

        df = df.assign(
            altitude=lambda x: (x.h / aero.ft).astype(int),
            vertical_rate=lambda x: (x.vs / aero.fpm).astype(int),
            groundspeed=lambda x: (x.v / aero.kts).astype(int),
            cas_const_cl=cas_const,
            mach_const_cl=mach_const,
            h_const_cas_start=h_const_cas,
            h_const_mach_start=h_const_mach,
            alt_cr=alt_cr,
        )

        return df

    def descent(self, **kwargs) -> pd.DataFrame:
        """Generate descent trajectory based on WRAP model.

        Args:
            **dt (int): Time step in seconds.
            **cas_const_de (int): Constant CAS for climb (kt).
            **mach_const_de (float): Constant Mach for climb (-).
            **alt_cr (int): top of descent altitude (ft).
            **random (bool): Generate trajectory with random parameters, default to False.
            **withcr (bool): Include a short cruise segment of 60 seconds, default to True.

        """
        dt = kwargs.get("dt", 1)
        random = kwargs.get("random", False)
        withcr = kwargs.get("withcr", True)

        a_lnd = self.wrap.landing_acceleration()["default"]
        v_app = self.wrap.finalapp_vcas()["default"]

        if random:
            alt_cr = kwargs.get(
                "alt_cr",
                self.rng.uniform(
                    self.wrap.cruise_alt()["minimum"],
                    self.wrap.cruise_alt()["maximum"],
                )
                * 1000
                / aero.ft,
            )

            mach_const = kwargs.get(
                "mach_const_de",
                self.rng.uniform(
                    self.wrap.descent_const_mach()["minimum"],
                    self.wrap.descent_const_mach()["maximum"],
                ),
            )

            cas_const = kwargs.get(
                "cas_const_de",
                self.rng.uniform(
                    self.wrap.descent_const_vcas()["minimum"],
                    self.wrap.descent_const_vcas()["maximum"],
                )
                / aero.kts,
            )

            vs_constmach = self.rng.uniform(
                self.wrap.descent_vs_conmach()["minimum"],
                self.wrap.descent_vs_conmach()["maximum"],
            )

            vs_constcas = self.rng.uniform(
                self.wrap.descent_vs_concas()["minimum"],
                self.wrap.descent_vs_concas()["maximum"],
            )

            vs_post_constcas = self.rng.uniform(
                self.wrap.descent_vs_post_concas()["minimum"],
                self.wrap.descent_vs_post_concas()["maximum"],
            )

        else:
            mach_const = kwargs.get(
                "mach_const_de", self.wrap.descent_const_mach()["default"]
            )
            cas_const = kwargs.get(
                "cas_const_de", self.wrap.descent_const_vcas()["default"] / aero.kts
            )
            alt_cr = kwargs.get(
                "alt_cr", self.wrap.cruise_alt()["default"] * 1000 / aero.ft
            )
            vs_constmach = self.wrap.descent_vs_conmach()["default"]
            vs_constcas = self.wrap.descent_vs_concas()["default"]
            vs_post_constcas = self.wrap.descent_vs_post_concas()["default"]

        vcas_const = cas_const * aero.kts
        alt_cr = np.round(alt_cr, -2)  # round cruise altitude to flight level
        h_cr = alt_cr * aero.ft
        vs_fa = self.wrap.finalapp_vs()["default"]
        h_const_cas = self.wrap.descent_cross_alt_concas()["default"] * 1000

        h_const_mach = aero.crossover_alt(vcas_const, mach_const)

        data = []

        # intitial conditions
        a = -0.2
        t = 0
        s = 0
        h = h_cr
        v = aero.mach2tas(mach_const, h_cr)
        vs = 0
        seg = None

        while True:
            data.append([t, h, s, v, vs, seg])
            t = t + dt
            s = s + v * dt
            h = h + vs * dt

            if t < 60 and withcr:
                v = aero.mach2tas(mach_const, h)
                vs = 0
                seg = "CR"
            elif h > h_const_mach:
                v = aero.mach2tas(mach_const, h)
                vs = vs_constmach
                seg = "MACH"
            elif h > h_const_cas:
                v = aero.cas2tas(vcas_const, h)
                vs = vs_constcas
                seg = "CAS"
            elif h > 1000 * aero.ft:
                v = v + a * dt
                if aero.tas2cas(v, h) < v_app:
                    v = aero.cas2tas(v_app, h)
                vs = vs_post_constcas
                seg = "POST-CAS"
            elif h > 0:
                v = v_app
                vs = vs_fa
                seg = "FA"
            else:
                h = 0
                vs = 0
                v = v + a_lnd * dt
                seg = "LD"

                if v <= 0:
                    break

        df = pd.DataFrame(data, columns=["t", "h", "s", "v", "vs", "seg"])

        if self.noise:
            n = df.shape[0]
            df = df.eval("""
            h = h + @self.rng.normal(0, @self.sigma_h, @n)
            s = s + @self.rng.normal(0, @self.sigma_s, @n)
            v = v + @self.rng.normal(0, @self.sigma_v, @n)
            vs = vs + @self.rng.normal(0, @self.sigma_vs, @n)
            """)

        df = df.assign(
            altitude=lambda x: (x.h / aero.ft).astype(int),
            vertical_rate=lambda x: (x.vs / aero.fpm).astype(int),
            groundspeed=lambda x: (x.v / aero.kts).astype(int),
            cas_const_de=cas_const,
            vcas_const_de=vcas_const,
            mach_const_de=mach_const,
            va_app=v_app,
            vs_constmach=vs_constmach,
            vs_constcas=vs_constcas,
            h_const_mach_end=h_const_mach,
            h_const_cas_end=h_const_cas,
            alt_cr=alt_cr,
        )

        return df

    def cruise(self, **kwargs) -> pd.DataFrame:
        """Generate descent trajectory based on WRAP model.

        Args:
            **dt (int): Time step in seconds.
            **range_cr (int): Cruise range (km).
            **alt_cr (int): Cruise altitude (ft).
            **mach_cr (float): Cruise Mach number (-).
            **random (bool): Generate trajectory with random parameters.

        """
        dt = kwargs.get("dt", 1)
        random = kwargs.get("random", False)

        if random:
            distance = kwargs.get(
                "range_cr",
                self.rng.uniform(
                    self.wrap.cruise_range()["minimum"],
                    self.wrap.cruise_range()["maximum"],
                )
                * 1000,
            )

            alt_cr = kwargs.get(
                "alt_cr",
                self.rng.uniform(
                    self.wrap.cruise_alt()["minimum"],
                    self.wrap.cruise_alt()["maximum"],
                )
                * 1000
                / aero.ft,
            )

            mach_cr = kwargs.get(
                "mach_cr",
                self.rng.uniform(
                    self.wrap.cruise_mach()["minimum"],
                    self.wrap.cruise_mach()["maximum"],
                ),
            )
        else:
            distance = kwargs.get(
                "range_cr", self.wrap.cruise_range()["default"] * 1000
            )
            mach_cr = kwargs.get("mach_cr", self.wrap.cruise_mach()["default"])
            alt_cr = kwargs.get(
                "alt_cr", self.wrap.cruise_alt()["default"] * 1000 / aero.ft
            )

        alt_cr = np.round(alt_cr, -2)  # round to flight level
        h_cr = alt_cr * aero.ft

        data = []

        # intitial conditions
        t = 0
        s = 0
        v = aero.mach2tas(mach_cr, h_cr)
        vs = 0

        while True:
            data.append([t, h_cr, s, v, vs])
            t = t + dt
            s = s + v * dt

            if s > distance:
                break

        df = pd.DataFrame(data, columns=["t", "h", "s", "v", "vs"])

        if self.noise:
            n = df.shape[0]
            df = df.eval("""
            h = h + @self.rng.normal(0, @self.sigma_h, @n)
            s = s + @self.rng.normal(0, @self.sigma_s, @n)
            v = v + @self.rng.normal(0, @self.sigma_v, @n)
            vs = vs + @self.rng.normal(0, @self.sigma_vs, @n)
            """)

        df = df.assign(
            altitude=lambda x: (x.h / aero.ft).astype(int),
            vertical_rate=lambda x: (x.vs / aero.fpm).astype(int),
            groundspeed=lambda x: (x.v / aero.kts).astype(int),
            alt_cr=alt_cr,
            mach_cr=mach_cr,
        )

        return df

    def complete(self, **kwargs) -> pd.DataFrame:
        """Generate a complete trajectory based on WRAP model.

        Args:
            **dt (int): Time step in seconds.
            **cas_const_cl (int): Constant CAS for climb (kt).
            **mach_const_cl (float): Constant Mach for climb (-).
            **cas_const_de (int): Constant CAS for climb (kt).
            **mach_const_de (float): Constant Mach for climb (-).
            **range_cr (int): Cruise range (km).
            **alt_cr (int): Target cruise altitude (ft).
            **mach_cr (float): Cruise Mach number (-).
            **random (bool): Generate trajectory with random parameters.

        """
        df_cr = self.cruise(**kwargs)

        if "alt_cr" not in kwargs:
            kwargs["alt_cr"] = df_cr["alt_cr"].iloc[0]

        df_cl = self.climb(mach_const_cl=df_cr["mach_cr"].iloc[0], **kwargs)

        df_de = self.descent(mach_const_de=df_cr["mach_cr"].iloc[0], **kwargs)

        df = pd.DataFrame(
            {
                "t": np.concatenate(
                    [
                        df_cl.t,
                        df_cl.t.iloc[-1] + df_cr.t,
                        df_cl.t.iloc[-1] + df_cr.t.iloc[-1] + df_de.t,
                    ]
                ),
                "h": np.concatenate([df_cl.h, df_cr.h, df_de.h]),
                "s": np.concatenate(
                    [
                        df_cl.s,
                        df_cl.s.iloc[-1] + df_cr.s,
                        df_cl.s.iloc[-1] + df_cr.s.iloc[-1] + df_de.s,
                    ]
                ),
                "v": np.concatenate([df_cl.v, df_cr.v, df_de.v]),
                "vs": np.concatenate([df_cl.vs, df_cr.vs, df_de.vs]),
            }
        )

        df = df.assign(
            altitude=lambda x: (x.h / aero.ft).astype(int),
            vertical_rate=lambda x: (x.vs / aero.fpm).astype(int),
            groundspeed=lambda x: (x.v / aero.kts).astype(int),
        )

        return df
