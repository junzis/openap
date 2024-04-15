"""Using fuzzy logic to indentify flight phase in trajectory data."""

import numpy as np
from matplotlib import pyplot as plt

from openap.extra import fuzzy


class FlightPhase(object):
    """Fuzzy logic flight phase identification."""

    def __init__(self):
        """Initialize of the FlightPhase object."""
        super(FlightPhase, self).__init__()

        # logic states
        self.alt_range = np.arange(0, 40000, 1)
        self.roc_range = np.arange(-4000, 4000, 0.1)
        self.spd_range = np.arange(0, 600, 1)
        self.states = np.arange(0, 6, 0.01)

        self.alt_gnd = fuzzy.zmf(self.alt_range, 0, 200)
        self.alt_lo = fuzzy.gaussmf(self.alt_range, 10000, 10000)
        self.alt_hi = fuzzy.gaussmf(self.alt_range, 35000, 20000)

        self.roc_zero = fuzzy.gaussmf(self.roc_range, 0, 100)
        self.roc_plus = fuzzy.smf(self.roc_range, 10, 1000)
        self.roc_minus = fuzzy.zmf(self.roc_range, -1000, -10)

        self.spd_hi = fuzzy.gaussmf(self.spd_range, 600, 100)
        self.spd_md = fuzzy.gaussmf(self.spd_range, 300, 100)
        self.spd_lo = fuzzy.gaussmf(self.spd_range, 0, 50)

        self.state_ground = fuzzy.gaussmf(self.states, 1, 0.1)
        self.state_climb = fuzzy.gaussmf(self.states, 2, 0.1)
        self.state_descent = fuzzy.gaussmf(self.states, 3, 0.1)
        self.state_cruise = fuzzy.gaussmf(self.states, 4, 0.1)
        self.state_level = fuzzy.gaussmf(self.states, 5, 0.1)

        self.state_lable_map = {1: "GND", 2: "CL", 3: "DE", 4: "CR", 5: "LVL"}

        self.ts = None
        self.alt = None
        self.spd = None
        self.roc = None

    def set_trajectory(self, ts, alt, spd, roc):
        """Set trajectory data.

        Args:
            ts (list): Time (unit: second).
            alt (list): Altitude (unit: ft).
            spd (list): True airspeed (unit: kt).
            roc (list): Rate of climb (unit: ft/min). Negative for descent.

        """
        self.ts = ts - ts[0]
        self.alt = alt
        self.spd = spd
        self.roc = roc

        if len(set([len(self.ts), len(self.alt), len(self.spd), len(self.roc)])) > 1:
            raise RuntimeError("Input lists must have same length.")

        self.ndata = len(self.ts)

        return

    def phaselabel(self, twindow=60):
        """Fuzzy logic for determining phase label.

        Args:
            twindow (int): Time window in number of seconds. Default to 60.

        Returns:
            list: Labels could be: ground [GND], climb [CL], descent [DE],
                cruise [CR], leveling [LVL].

        """
        if self.ts is None:
            raise RuntimeError(
                "Trajectory data not set, run set_trajectory(ts, alt, spd, roc) first"
            )

        idxs = np.arange(0, self.ndata)

        labels = ["NA"] * self.ndata

        twindows = self.ts // twindow

        for tw in range(0, int(max(twindows))):
            if tw not in twindows:
                continue

            mask = twindows == tw

            idxchk = idxs[mask]
            altchk = self.alt[mask]
            spdchk = self.spd[mask]
            rocchk = self.roc[mask]

            # mean value or extream value as range
            alt = max(min(np.mean(altchk), self.alt_range[-1]), self.alt_range[0])
            spd = max(min(np.mean(spdchk), self.spd_range[-1]), self.spd_range[0])
            roc = max(min(np.mean(rocchk), self.roc_range[-1]), self.roc_range[0])

            # make sure values are within the boundaries
            alt = max(min(alt, self.alt_range[-1]), self.alt_range[0])
            spd = max(min(spd, self.spd_range[-1]), self.spd_range[0])
            roc = max(min(roc, self.roc_range[-1]), self.roc_range[0])

            alt_level_gnd = fuzzy.interp_membership(self.alt_range, self.alt_gnd, alt)
            alt_level_lo = fuzzy.interp_membership(self.alt_range, self.alt_lo, alt)
            alt_level_hi = fuzzy.interp_membership(self.alt_range, self.alt_hi, alt)

            spd_level_hi = fuzzy.interp_membership(self.spd_range, self.spd_hi, spd)
            spd_level_md = fuzzy.interp_membership(self.spd_range, self.spd_md, spd)
            spd_level_lo = fuzzy.interp_membership(self.spd_range, self.spd_lo, spd)

            roc_level_zero = fuzzy.interp_membership(self.roc_range, self.roc_zero, roc)
            roc_level_plus = fuzzy.interp_membership(self.roc_range, self.roc_plus, roc)
            roc_level_minus = fuzzy.interp_membership(
                self.roc_range, self.roc_minus, roc
            )

            # print alt_level_gnd, alt_level_lo, alt_level_hi
            # print roc_level_zero, roc_level_plus, roc_level_minus
            # print spd_level_hi, spd_level_md, spd_level_lo

            rule_ground = min(alt_level_gnd, roc_level_zero, spd_level_lo)
            state_activate_ground = np.fmin(rule_ground, self.state_ground)

            rule_climb = min(alt_level_lo, roc_level_plus, spd_level_md)
            state_activate_climb = np.fmin(rule_climb, self.state_climb)

            rule_descent = min(alt_level_lo, roc_level_minus, spd_level_md)
            state_activate_descent = np.fmin(rule_descent, self.state_descent)

            rule_cruise = min(alt_level_hi, roc_level_zero, spd_level_hi)
            state_activate_cruise = np.fmin(rule_cruise, self.state_cruise)

            rule_level = min(alt_level_lo, roc_level_zero, spd_level_md)
            state_activate_level = np.fmin(rule_level, self.state_level)

            aggregated = np.max(
                np.vstack(
                    [
                        state_activate_ground,
                        state_activate_climb,
                        state_activate_descent,
                        state_activate_cruise,
                        state_activate_level,
                    ]
                ),
                axis=0,
            )

            state_raw = fuzzy.defuzz(self.states, aggregated, "lom")
            state = int(round(state_raw))
            if state > 6:
                state = 6
            if state < 1:
                state = 1

            if len(idxchk) > 0:
                label = self.state_lable_map[state]
                labels[idxchk[0] : (idxchk[-1] + 1)] = [label] * len(idxchk)

        return labels

    def plot_logics(self):
        """Visualize fuzzy logic membership functions."""
        plt.figure(figsize=(10, 8))

        plt.subplot(411)
        plt.plot(self.alt_range, self.alt_gnd, lw=2, label="Ground")
        plt.plot(self.alt_range, self.alt_lo, lw=2, label="Low")
        plt.plot(self.alt_range, self.alt_hi, lw=2, label="High")
        plt.ylim([-0.05, 1.05])
        plt.ylabel("Altitude (ft)")
        plt.yticks([0, 1])
        plt.legend()

        plt.subplot(412)
        plt.plot(self.roc_range, self.roc_zero, lw=2, label="Zero")
        plt.plot(self.roc_range, self.roc_plus, lw=2, label="Positive")
        plt.plot(self.roc_range, self.roc_minus, lw=2, label="Negative")
        plt.ylim([-0.05, 1.05])
        plt.ylabel("RoC (ft/m)")
        plt.yticks([0, 1])
        plt.legend()

        plt.subplot(413)
        plt.plot(self.spd_range, self.spd_hi, lw=2, label="High")
        plt.plot(self.spd_range, self.spd_md, lw=2, label="Midium")
        plt.plot(self.spd_range, self.spd_lo, lw=2, label="Low")
        plt.ylim([-0.05, 1.05])
        plt.ylabel("Speed (kt)")
        plt.yticks([0, 1])
        plt.legend()

        plt.subplot(414)
        plt.plot(self.states, self.state_ground, lw=2, label="ground")
        plt.plot(self.states, self.state_climb, lw=2, label="climb")
        plt.plot(self.states, self.state_descent, lw=2, label="descent")
        plt.plot(self.states, self.state_cruise, lw=2, label="cruise")
        plt.plot(self.states, self.state_level, lw=2, label="level flight")
        plt.ylim([-0.05, 1.05])
        plt.ylabel("Flight Phases")
        plt.yticks([0, 1])
        plt.legend(prop={"size": 7})
        plt.show()

    def _get_to_ic(self):
        # get the data chunk up to certain ft
        istart = 0
        iend = 0
        for i in range(0, self.ndata):
            if self.alt[i] < 1500:  # ft
                iend = i
                continue
            else:
                break

        # keep only the chunk in taking-off states, break at starting point
        spdtmp = self.spd[iend]
        for i in reversed(list(range(0, iend))):
            if self.spd[i] < 30 and self.spd[i] > spdtmp:
                break
            elif self.spd[i] < 5:
                break
            else:
                istart = i
                spdtmp = self.spd[i]

        # ignore too long take-off
        if self.ts[iend] - self.ts[istart] > 300:
            return None

        # ignore insufficient chunk size
        if iend - istart < 10:
            return None

        # ignore no in air data
        if self.alt[iend] < 200:
            return None

        # find the liftoff moment
        ilof = istart
        for i in range(istart + 1, iend):
            if abs(self.alt[i] - self.alt[i - 1]) > 10:
                ilof = i
                break

        # not sufficient data
        if ilof - istart < 5:
            return None

        return (istart, ilof, iend + 1)

    def _get_fa_ld(self):

        # get the approach + landing data chunk (h=0)
        istart = 0
        iend = 0

        for i in reversed(list(range(0, self.ndata))):
            if self.alt[i] < 1500:  # ft
                istart = i
            else:
                break

        # keep only the chunk in landing deceleration states, break at taxing point
        spdtmp = self.spd[istart]
        for i in range(istart, self.ndata):
            if self.spd[i] <= 50 and self.spd[i] >= spdtmp:  # kts
                break
            elif self.spd[i] < 30:
                break
            else:
                iend = i
                spdtmp = self.spd[i]

        # ignore insufficient chunk size
        # if iend - istart < 20:
        #     return None

        # ignore QNH altitude, or no in-air data
        if self.alt[istart] < 100:
            return None

        # # ignore where the end speed too high, ie. not breaked
        # if spd[iend] > 60: # kts
        #     return None

        # find the landing moment
        ild = iend
        for i in reversed(list(range(istart, iend - 1))):
            if abs(self.alt[i] - self.alt[i + 1]) > 10:
                ild = i
                break

        # ignore ground or air data sample less than 4
        if ild - istart < 5 or iend - ild < 5:
            return None

        return (istart, ild, iend + 1)

    def _get_cl(self):
        labels = np.array(self.phaselabel())

        if "CL" not in labels:
            return None

        idx = np.where(np.array(labels) == "CL")[0]

        istart = idx[0]
        iend = idx[-1]

        return istart, iend

    def _get_de(self):

        labels = np.array(self.phaselabel())

        if "DE" not in labels:
            return None

        idx = np.where(np.array(labels) == "DE")[0]

        istart = idx[0]
        iend = idx[-1]

        if "LVL" in labels[istart:iend]:
            isCDA = False
        else:
            isCDA = True

        return istart, iend, isCDA

    def _get_cr(self):
        # CR start = CL end, CR end = DE start
        ttCL = self._get_cl()

        if not ttCL:
            return None

        ttDE = self._get_de()

        if not ttDE:
            return None

        ttDE = ttDE[0:2]

        istart = ttCL[-1]
        iend = ttDE[0]

        if iend - istart < 200:
            # too few samples
            return None

        return istart, iend

    def flight_phase_indices(self):
        """Get the indices of data, of which different flight phase start.

        Returns:
            dict: Indices for takeoff (TO), initial climb (IC), climb (CL),
                cruise (CR), descent (DE), final approach (FA), landing (LD).

        """
        # Process the data and get the phase index
        ii_toic = self._get_to_ic()
        ii_cl = self._get_cl()
        ii_de = self._get_de()
        ii_fald = self._get_fa_ld()

        ito = ii_toic[0] if ii_toic is not None else None
        iic = ii_toic[1] if ii_toic is not None else None

        if ii_toic is not None:
            icl = ii_toic[2]
        else:
            if ii_cl is not None:
                icl = ii_cl[0]
            else:
                icl = None

        icr = ii_cl[1] if ii_cl is not None else None

        ide = ii_de[0] if ii_de is not None else None

        if ii_fald is not None:
            ifa = ii_fald[0]
            ild = ii_fald[1]
            iend = ii_fald[2]
        elif ii_de is not None:
            ifa = ii_de[1]
            ild = None
            iend = len(self.ts)
        else:
            ifa = None
            ild = None
            iend = len(self.ts)

        idx = {
            "TO": ito,
            "IC": iic,
            "CL": icl,
            "CR": icr,
            "DE": ide,
            "FA": ifa,
            "LD": ild,
            "END": iend,
        }

        return idx
