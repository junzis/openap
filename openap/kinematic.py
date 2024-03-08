"""OpenAP kinematic model.

This module implements functions to access the WRAP kinematic model.

Examples:
    WRAP model can be accessed as follows::

        from openap import WRAP
        wrap = WRAP(ac='A320')
        wrap.cruise_mach()

    with result::

        {
            'default': 0.78,
            'minimum': 0.75,
            'maximum': 0.8,
            'statmodel': 'beta',
            'statmodel_params': [17.82, 5.05, 0.62, 0.2]
        }

"""
import os
import glob
import pandas as pd

curr_path = os.path.dirname(os.path.realpath(__file__))
dir_wrap = os.path.join(curr_path, "/data/wrap/")
file_synonym = os.path.join(curr_path, "/data/wrap/_synonym.csv")

wrap_synonym = pd.read_csv(file_synonym)


class WRAP(object):
    """Construct the kinematic model of the aicraft."""

    def __init__(self, ac, **kwargs):
        """Initialize WRAP object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).

        """
        super(WRAP, self).__init__()

        self.ac = ac.lower()

        self.use_synonym = kwargs.get("use_synonym", False)

        wrap_files = glob.glob(dir_wrap + "*.txt")
        ac_wrap_available = [s[-8:-4].lower() for s in wrap_files]

        if self.ac not in ac_wrap_available and not self.use_synonym:
            raise ValueError((f"Kinematic model for {self.ac} not"
                                " available in OpenAP."))

        if self.ac not in ac_wrap_available and self.use_synonym:
            syno = wrap_synonym.query("orig==@self.ac")
            if syno.shape[0] > 0:
                self.ac = syno.new.iloc[0]
            else:
                raise ValueError(
                    f"Kinematic model for {self.ac} not available in OpenAP."
                )

        self.df = pd.read_fwf(os.path.join(dir_wrap, self.ac + ".txt"))

    def _get_var(self, var):
        r = self.df[self.df["variable"] == var]

        if r.shape[0] == 0:
            raise ValueError(f"Variable {var} not found")

        v = r.values[0]

        res = {
            "default": v[3],
            "minimum": v[4],
            "maximum": v[5],
            "statmodel": v[6],
            "statmodel_params": [float(i) for i in v[7].split("|")],
        }
        return res

    def takeoff_speed(self):
        """Get takeoff speed."""
        return self._get_var("to_v_lof")

    def takeoff_distance(self):
        """Get takeoff takeoff distance."""
        return self._get_var("to_d_tof")

    def takeoff_acceleration(self):
        """Get takeoff takeoff acceleration."""
        return self._get_var("to_acc_tof")

    def initclimb_vcas(self):
        """Get initial climb CAS."""
        return self._get_var("ic_va_avg")

    def initclimb_vs(self):
        """Get initial climb vertical rate."""
        return self._get_var("ic_vs_avg")

    def climb_range(self):
        """Get climb range distance (in km)."""
        return self._get_var("cl_d_range")

    def climb_const_vcas(self):
        """Get speed for constant CAS climb."""
        return self._get_var("cl_v_cas_const")

    def climb_const_mach(self):
        """Get speed during constant Mach climb."""
        return self._get_var("cl_v_mach_const")

    def climb_cross_alt_concas(self):
        """Get cross over altitude when constant CAS climb starts."""
        return self._get_var("cl_h_cas_const")

    def climb_cross_alt_conmach(self):
        """Get cross over altitude from constant CAS to Mach climb."""
        return self._get_var("cl_h_mach_const")

    def climb_vs_pre_concas(self):
        """Get vertical rate before constant CAS climb."""
        return self._get_var("cl_vs_avg_pre_cas")

    def climb_vs_concas(self):
        """Get vertical rate during constant CAS climb."""
        return self._get_var("cl_vs_avg_cas_const")

    def climb_vs_conmach(self):
        """Get vertical rate during constant Mach climb."""
        return self._get_var("cl_vs_avg_mach_const")

    def cruise_range(self):
        """Get crusie range."""
        return self._get_var("cr_d_range")

    def cruise_alt(self):
        """Get average cruise altitude."""
        return self._get_var("cr_h_mean")

    def cruise_init_alt(self):
        """Get initial crusie altitude."""
        return self._get_var("cr_h_init")

    def cruise_mach(self):
        """Get average crusie Mach number."""
        return self._get_var("cr_v_mach_mean")

    def descent_range(self):
        """Get descent range."""
        return self._get_var("de_d_range")

    def descent_const_mach(self):
        """Get speed during the constant Mach descent."""
        return self._get_var("de_v_mach_const")

    def descent_const_vcas(self):
        """Get speed during the constant CAS descent."""
        return self._get_var("de_v_cas_const")

    def descent_cross_alt_conmach(self):
        """Get crossover altitude from constant Mach to CAS descent."""
        return self._get_var("de_h_mach_const")

    def descent_cross_alt_concas(self):
        """Get crossover altitude from constant Mach to CAS descent."""
        return self._get_var("de_h_cas_const")

    def descent_vs_conmach(self):
        """Get vertical rate during constant Mach descent."""
        return self._get_var("de_vs_avg_mach_const")

    def descent_vs_concas(self):
        """Get vertical rate during constant CAS descent."""
        return self._get_var("de_vs_avg_cas_const")

    def descent_vs_post_concas(self):
        """Get vertical rate after constant CAS descent."""
        return self._get_var("de_vs_avg_after_cas")

    def finalapp_vcas(self):
        """Get CAS for final approach."""
        return self._get_var("fa_va_avg")

    def finalapp_vs(self):
        """Get vertical speed for final approach."""
        return self._get_var("fa_vs_avg")

    def landing_speed(self):
        """Get landing speed."""
        return self._get_var("ld_v_app")

    def landing_distance(self):
        """Get breaking distance for landing."""
        return self._get_var("ld_d_brk")

    def landing_acceleration(self):
        """Get landing deceleration."""
        return self._get_var("ld_acc_brk")
