"""Trajectory generator based on WRAP model.

This module defines the Generator class than can be used to generate flight
trajectories using the WRAP kinematic performance model from OpenAP.

Examples::

    trajgen = Generator(ac='a320')

    trajgen.enable_noise()   # enable Gaussian noise in trajectory data

    data_cl = trajgen.climb(dt=10, cas_const_cl=280, mach_const_cl=0.78, alt_cr=35000)
    data_cl = trajgen.climb(dt=10, random=True)  # using radom paramerters

    data_de = trajgen.descent(dt=10, cas_const_de=280, mach_const_de=0.78, alt_cr=35000)
    data_de = trajgen.descent(dt=10, random=True)

    data_cr = trajgen.cruise(dt=60, range_cr=2000, alt_cr=35000, m_cr=0.78)
    data_cr = trajgen.cruise(dt=60, random=True)

    data_all = trajgen.complete(dt=10, alt_cr=35000, m_cr=0.78,
                                cas_const_cl=280, mach_const_cl=0.78,
                                cas_const_de=280, mach_const_de=0.78)
    data_all = trajgen.complete(dt=10, random=True)

"""

from openap import aero, prop, WRAP
import numpy as np


class Generator(object):
    """Generate trajectory using WRAP model."""

    def __init__(self, ac, eng=None):
        """Intitialize the generator.

        Args:
            ac (string): Aircraft type.
            eng (string): Engine type. Leave empty for default engine type in OpenAP.

        Returns:
            dict: flight trajectory

        """
        super(Generator, self).__init__()

        self.ac = ac
        self.acdict = prop.aircraft(self.ac)

        if eng is None:
            self.eng = self.acdict["engine"]["default"]
        else:
            self.eng = eng
        self.engdict = prop.engine(self.eng)

        self.wrap = WRAP(self.ac)
        # self.thrust = Thrust(self.ac, self.eng)
        # self.drag = Drag(self.ac)
        # self.fuelflow = Thrust(self.ac, self.eng)

        # for noise generation
        self.sigma_v = 0
        self.sigma_vs = 0
        self.sigma_h = 0
        self.sigma_s = 0

    def enable_noise(self):
        """Adding noise to the generated trajectory.

        The noise model is based on ADS-B Version 1&2, NACv=3 and NACp=10
        """
        self.sigma_v = 0.5
        self.sigma_vs = 0.75
        self.sigma_h = 7.5
        self.sigma_s = 5

    def climb(self, **kwargs):
        """Generate climb trajectory based on WRAP model.

        Args:
            **dt (int): Time step in seconds.
            **cas_const_cl (int): Constant CAS for climb (kt).
            **mach_const_cl (float): Constant Mach for climb (-).
            **alt_cr (int): Target cruise altitude (ft).
            **random (bool): Generate trajectory with random paramerters.

        Returns:
            dict: Flight trajectory.

        """
        dt = kwargs.get("dt", 1)
        random = kwargs.get("random", False)

        a_tof = self.wrap.takeoff_acceleration()["default"]
        v_tof = self.wrap.takeoff_speed()["default"]

        if random:
            cas_const = kwargs.get(
                "cas_const_cl",
                np.random.uniform(
                    self.wrap.climb_const_vcas()["minimum"],
                    self.wrap.climb_const_vcas()["maximum"],
                )
                / aero.kts,
            )

            mach_const = kwargs.get(
                "mach_const_cl",
                np.random.uniform(
                    self.wrap.climb_const_mach()["minimum"],
                    self.wrap.climb_const_mach()["maximum"],
                ),
            )

            alt_cr = kwargs.get(
                "alt_cr",
                np.random.uniform(
                    self.wrap.cruise_alt()["minimum"], self.wrap.cruise_alt()["maximum"]
                )
                * 1000
                / aero.ft,
            )

            vs_pre_constcas = np.random.uniform(
                self.wrap.climb_vs_pre_concas()["minimum"],
                self.wrap.climb_vs_pre_concas()["maximum"],
            )

            vs_constcas = np.random.uniform(
                self.wrap.climb_vs_concas()["minimum"],
                self.wrap.climb_vs_concas()["maximum"],
            )

            vs_constmach = np.random.uniform(
                self.wrap.climb_vs_conmach()["minimum"],
                self.wrap.climb_vs_conmach()["maximum"],
            )

        else:
            cas_const = kwargs.get(
                "cas_const_cl", self.wrap.climb_const_vcas()["default"] / aero.kts
            )
            cas_const = max(
                cas_const, v_tof / aero.kts
            )  # cas can not be smaller then takeoff speed
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
        alt_cr = np.round(alt_cr, -2)  # round cruise altitude to flight level
        h_cr = alt_cr * aero.ft  # meters
        vs_ic = self.wrap.initclimb_vs()["default"]
        h_const_cas = self.wrap.climb_cross_alt_concas()["default"] * 1000

        h_const_mach = aero.crossover_alt(vcas_const, mach_const)
        if h_const_mach > h_cr:
            print(
                "Warning: const mach crossover altitude higher than cruise altitude, altitude clipped."
            )

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

        data = np.array(data)
        ndata = len(data)
        datadict = {
            "t": data[:, 0],  # t, seconds
            "h": data[:, 1] + np.random.normal(0, self.sigma_h, ndata),  # h, m
            "s": data[:, 2] + np.random.normal(0, self.sigma_s, ndata),  # s, m
            "v": data[:, 3] + np.random.normal(0, self.sigma_v, ndata),  # v, m/s
            "vs": data[:, 4] + np.random.normal(0, self.sigma_vs, ndata),  # VS, m/s
            "seg": data[:, 5],
            "cas_const_cl": cas_const,
            "mach_const_cl": mach_const,
            "h_const_cas_start": h_const_cas,
            "h_const_mach_start": h_const_mach,
            "alt_cr": alt_cr,  # alt_cr, ft
        }

        return datadict

    def descent(self, **kwargs):
        """Generate descent trajectory based on WRAP model.

        Args:
            **dt (int): Time step in seconds.
            **cas_const_de (int): Constant CAS for climb (kt).
            **mach_const_de (float): Constant Mach for climb (-).
            **alt_cr (int): Target cruise altitude (ft).
            **random (bool): Generate trajectory with random paramerters.

        Returns:
            dict: Flight trajectory.

        """
        dt = kwargs.get("dt", 1)
        random = kwargs.get("random", False)
        withcr = kwargs.get("withcr", True)

        a_lnd = self.wrap.landing_acceleration()["default"]
        v_app = self.wrap.finalapp_vcas()["default"]

        if random:
            alt_cr = kwargs.get(
                "alt_cr",
                np.random.uniform(
                    self.wrap.cruise_alt()["minimum"], self.wrap.cruise_alt()["maximum"]
                )
                * 1000
                / aero.ft,
            )

            mach_const = kwargs.get(
                "mach_const_de",
                np.random.uniform(
                    self.wrap.descent_const_mach()["minimum"],
                    self.wrap.descent_const_mach()["maximum"],
                ),
            )

            cas_const = kwargs.get(
                "cas_const_de",
                np.random.uniform(
                    self.wrap.descent_const_vcas()["minimum"],
                    self.wrap.descent_const_vcas()["maximum"],
                )
                / aero.kts,
            )

            vs_constmach = np.random.uniform(
                self.wrap.descent_vs_conmach()["minimum"],
                self.wrap.descent_vs_conmach()["maximum"],
            )

            vs_constcas = np.random.uniform(
                self.wrap.descent_vs_concas()["minimum"],
                self.wrap.descent_vs_concas()["maximum"],
            )

            vs_post_constcas = np.random.uniform(
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
        if h_const_mach > h_cr:
            print(
                "Warning: const mach crossover altitude higher than cruise altitude, altitude clipped."
            )

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

        data = np.array(data)
        ndata = len(data)
        datadict = {
            "t": data[:, 0],
            "h": data[:, 1] + np.random.normal(0, self.sigma_h, ndata),
            "s": data[:, 2] + np.random.normal(0, self.sigma_s, ndata),
            "v": data[:, 3] + np.random.normal(0, self.sigma_v, ndata),
            "vs": data[:, 4] + np.random.normal(0, self.sigma_vs, ndata),
            "seg": data[:, 5],
            "cas_const_de": cas_const,
            "vcas_const_de": vcas_const,
            "mach_const_de": mach_const,
            "v_app": v_app,
            "vs_constmach": vs_constmach,
            "vs_constcas": vs_constcas,
            "vs_post_constcas": vs_post_constcas,
            "h_const_mach_end": h_const_mach,
            "h_const_cas_end": h_const_cas,
            "alt_cr": alt_cr,
        }

        return datadict

    def cruise(self, **kwargs):
        """Generate descent trajectory based on WRAP model.

        Args:
            **dt (int): Time step in seconds.
            **range_cr (int): Cruise range (km).
            **alt_cr (int): Cruise altitude (ft).
            **mach_cr (float): Cruise Mach number (-).
            **random (bool): Generate trajectory with random paramerters.

        Returns:
            dict: flight trajectory

        """
        dt = kwargs.get("dt", 1)
        random = kwargs.get("random", False)

        if random:
            range = kwargs.get(
                "range_cr",
                np.random.uniform(
                    self.wrap.cruise_range()["minimum"],
                    self.wrap.cruise_range()["maximum"],
                )
                * 1000,
            )

            alt_cr = kwargs.get(
                "alt_cr",
                np.random.uniform(
                    self.wrap.cruise_alt()["minimum"], self.wrap.cruise_alt()["maximum"]
                )
                * 1000
                / aero.ft,
            )

            mach_cr = kwargs.get(
                "mach_cr",
                np.random.uniform(
                    self.wrap.cruise_mach()["minimum"],
                    self.wrap.cruise_mach()["maximum"],
                ),
            )
        else:
            range = kwargs.get("range_cr", self.wrap.cruise_range()["default"] * 1000)
            mach_cr = kwargs.get("mach_cr", self.wrap.cruise_mach()["default"])
            alt_cr = kwargs.get(
                "alt_cr", self.wrap.cruise_alt()["default"] * 1000 / aero.ft
            )

        alt_cr = np.round(alt_cr, -2)  # round cruise altitude to flight level
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

            if s > range:
                break

        data = np.array(data)
        ndata = len(data)
        datadict = {
            "t": data[:, 0],
            "h": data[:, 1] + np.random.normal(0, self.sigma_h, ndata),
            "s": data[:, 2] + np.random.normal(0, self.sigma_s, ndata),
            "v": data[:, 3] + np.random.normal(0, self.sigma_v, ndata),
            "vs": data[:, 4] + np.random.normal(0, self.sigma_vs, ndata),
            "alt_cr": alt_cr,
            "mach_cr": mach_cr,
        }

        return datadict

    def complete(self, **kwargs):
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
            **random (bool): Generate trajectory with random paramerters.
        Returns:
            dict: Flight trajectory.

        """
        data_cr = self.cruise(**kwargs)

        if "alt_cr" not in kwargs:
            kwargs["alt_cr"] = data_cr["alt_cr"]

        data_cl = self.climb(mach_const_cl=data_cr["mach_cr"], **kwargs)

        data_de = self.descent(mach_const_de=data_cr["mach_cr"], **kwargs)

        data = {
            "t": np.concatenate(
                [
                    data_cl["t"],
                    data_cl["t"][-1] + data_cr["t"],
                    data_cl["t"][-1] + data_cr["t"][-1] + data_de["t"],
                ]
            ),
            "h": np.concatenate([data_cl["h"], data_cr["h"], data_de["h"]]),
            "s": np.concatenate(
                [
                    data_cl["s"],
                    data_cl["s"][-1] + data_cr["s"],
                    data_cl["s"][-1] + data_cr["s"][-1] + data_de["s"],
                ]
            ),
            "v": np.concatenate([data_cl["v"], data_cr["v"], data_de["v"]]),
            "vs": np.concatenate([data_cl["vs"], data_cr["vs"], data_de["vs"]]),
        }
        return data
