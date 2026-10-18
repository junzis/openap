# %%
import re
from pathlib import Path
from typing import Optional

from numpy import ndarray

from .. import base, prop
from ..extra import ndarrayconvert


# %%
def bada3_aircraft(ac: str, bada_path: str) -> dict:
    """Update aircraft with BADA3 data.

    Args:
        ac: Aircraft identifier (e.g. "A320").
        bada_path: Path to the BADA3 data folder.

    Returns:
        Aircraft performance data from aircraft yaml file and BADA3.

    """

    def deep_update(target: dict, updates: dict):
        # recursively update values in target dictionary
        for key, value in updates.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                deep_update(target[key], value)
            else:
                target[key] = value

    aircraft = prop.aircraft(ac)
    deep_update(aircraft, load_bada3(ac, bada_path))
    return aircraft


def load_bada3(ac: str, bada_path: str) -> dict:
    """Load BADA3 aircraft performance data.

    Resolves the synonym and parses the associated .OPF file.

    Args:
        ac: Aircraft identifier (e.g. "A320").
        bada_path: Path to the BADA3 data folder.

    Returns:
        Full aircraft performance data including aerodynamic coefficients,
        thrust/fuel parameters, and metadata.

    Raises:
        FileNotFoundError: If the OPF file cannot be found.
        ValueError: If the aircraft code is not found in the SYNONYM file.

    """
    synonym_data = read_synonym(ac, bada_path)
    synonym = synonym_data["synonym"]
    file = Path(bada_path) / f"{synonym}__.OPF"

    if not file.exists():
        raise FileNotFoundError(f"OPF file not found: {file}")

    with file.open("r", encoding="utf-8") as f:
        content = f.read()

    opf_data = parse_opf_content(content)
    return {
        "typecode": ac,
        **synonym_data,
        **opf_data,
    }


def read_synonym(ac: str, bada_path: str, synonym_file: str = "SYNONYM.NEW") -> dict:
    """Parse the BADA3 SYNONYM file for aircraft information.

    Args:
        ac: Aircraft identifier (e.g. "A320").
        bada_path: Path to BADA3 folder.
        synonym_file: Name of the SYNONYM file. Defaults to "SYNONYM.NEW".

    Returns:
        Dictionary with keys 'manufacturer', 'model', and 'synonym'.

    Raises:
        FileNotFoundError: If the SYNONYM file is not found.
        ValueError: If the aircraft code is not found in the file.

    """

    file = Path(bada_path) / synonym_file

    if not file.exists():
        raise FileNotFoundError(f"SYNONYM file not found at: {file}")

    pattern = re.compile(
        r"^CD\s+[^\s]\s+(\S+)\s+"  # A/C code
        r"(\S(?:.*?\S)?)\s{2,}"  # Manufacturer
        r"(.+?)\s{2,}"  # Model
        r"(\S+)\s+.*?/$"  # Synonym
    )

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                code, manufacturer, model, synonym = match.groups()
                if code.upper() == ac.upper():
                    return {
                        "aircraft": f"{manufacturer.strip()} {model.strip()}",
                        "manufacturer": manufacturer.strip(),
                        "model": model.strip(),
                        "synonym": synonym.strip()[:4],
                    }

    raise ValueError(f"Aircraft code '{ac}' not found in SYNONYM file.")


def parse_opf_content(content: str) -> dict:
    """Parse a BADA3 .OPF file and extract aircraft performance parameters.

    Args:
        content: Raw text content of the .OPF file.

    Returns:
        Parsed aerodynamic and engine parameters including CD0/CD2 by phase,
        thrust coefficients, fuel flow coefficients, wing area, and engine type.

    """

    def get_drag_coeff(phase):
        # get drag coefficients
        pattern = (
            rf"CD\s+\d+\s+{phase}\s+\S+\s+[\d.E+-]+\s+" r"([\d.E+-]+)\s+([\d.E+-]+)"
        )
        m = re.search(pattern, content)
        return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)

    def get_floats(pattern, count, default=0.0):
        # get a certain number of floats from a regex pattern
        m = re.search(pattern, content, re.DOTALL)
        return (
            [float(m.group(i)) for i in range(1, count + 1)] if m else [default] * count
        )

    # actype
    engine_type = "turbofan"
    num_engines = None  # or default like 0
    m_actype = re.search(r"Actype[\s\S]*?^CD\s+.*$", content, re.MULTILINE)
    if m_actype:
        cd_line = m_actype.group(0)

        # Engine type
        if re.search(r"Turboprop", cd_line, re.IGNORECASE):
            engine_type = "turboprop"
        elif re.search(r"Jet", cd_line, re.IGNORECASE):
            engine_type = "turbofan"
        elif re.search(r"Piston", cd_line, re.IGNORECASE):
            engine_type = "piston"
        elif re.search(r"Electric", cd_line, re.IGNORECASE):
            engine_type = "electric"

        # Number of engines
        m_engines = re.search(r"(\d+)\s+engines?", cd_line, re.IGNORECASE)
        if m_engines:
            num_engines = int(m_engines.group(1))

    # aircraft mass
    mass_vals = get_floats(
        r"Mass \(t\).*?CD\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)"
        r"\s+([\d.E+-]+)\s+[\d.E+-]+",
        4,
    )

    # flight envelope
    envelope_vals = get_floats(
        r"Flight envelope.*?CD\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)"
        r"\s+[\d.E+-]+\s+[\d.E+-]+",
        3,
    )

    # wing area
    wing_area = get_floats(r"Wing Area.*?CD \d+\s+([\d.E+-]+)", 1)[0]

    # Drag coefficients
    cd0_cr, cd2_cr = get_drag_coeff("CR")
    cd0_ic, cd2_ic = get_drag_coeff("IC")
    cd0_to, cd2_to = get_drag_coeff("TO")
    cd0_ap, cd2_ap = get_drag_coeff("AP")
    cd0_ld, cd2_ld = get_drag_coeff("LD")

    # Landing gear
    cd0_lgear = get_floats(r"CD \d+\s+DOWN\s+([\d.E+-]+)", 1)[0]

    # Thrust coefficients
    ct = get_floats(
        r"Max climb thrust coefficients.*?CD\s+([\d.E+-]+)\s+([\d.E+-]+)"
        r"\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)",
        5,
    )
    ctdes = get_floats(
        r"Desc\(low\).*?CD\s+([\d.E+-]+)\s+([\d.E+-]+)"
        r"\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)",
        5,
    )

    # Fuel coefficients
    cf = get_floats(
        r"Thrust Specific Fuel Consumption Coefficients.*?CD\s+([\d.E+-]+)"
        r"\s+([\d.E+-]+)",
        2,
    )
    cfdes = get_floats(
        r"Descent Fuel Flow Coefficients.*?CD\s+([\d.E+-]+)\s+([\d.E+-]+)", 2
    )
    cfcr = get_floats(r"Cruise Corr.*?CD\s+([\d.E+-]+)", 1, default=1.0)[0]

    # Altitude for hpdes (from Desc level)
    hpdes = get_floats(
        r"Desc level.*?CD\s+[\d.E+-]+\s+[\d.E+-]+\s+([\d.E+-]+)",
        1,
        default=8000.0,
    )[0]

    ground_vals = get_floats(
        r"Ground.*?CD\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)"
        r"\s+([\d.E+-]+)\s+[\d.E+-]+",
        4,
    )

    # return as openap aircraft style dictionary
    return {
        "mtow": mass_vals[2] * 1e3,  # convert to kg
        "mref": mass_vals[0] * 1e3,
        "oew": mass_vals[1] * 1e3,
        "mpl": mass_vals[3] * 1e3,
        "vmo": envelope_vals[0],  # kt
        "mmo": envelope_vals[1],
        "ceiling": envelope_vals[2] * 0.3048,  # convert to m
        "fuselage": {
            "length": ground_vals[3],
        },
        "wing": {
            "area": wing_area,
            "span": ground_vals[2],
        },
        "engine": {
            "type": engine_type,
            "number": num_engines,
        },
        "CD0": {"CR": cd0_cr, "IC": cd0_ic, "TO": cd0_to, "AP": cd0_ap, "LD": cd0_ld},
        "CD2": {"CR": cd2_cr, "IC": cd2_ic, "TO": cd2_to, "AP": cd2_ap, "LD": cd2_ld},
        "CD0_lgear": cd0_lgear,
        "Ct": ct,
        "CTdeshigh": ctdes[1],
        "CTdeslow": ctdes[0],
        "CTdesapp": ctdes[3],
        "CTdesld": ctdes[4],
        "HpDes": hpdes,
        "Cf": cf,
        "CfDes": cfdes,
        "CfCrz": cfcr,
    }


# %%
class Drag(base.DragBase):
    """Compute the drag of an aircraft using BADA3 models."""

    def __init__(self, ac: str, bada_path: str, model=None, **kwargs):
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()
        self.smooth = kwargs.get("smooth", False)
        if not model:
            model = load_bada3(ac, bada_path)
        self.S = model["wing"]["area"]
        self.cd0_cr = model["CD0"]["CR"]
        self.cd2_cr = model["CD2"]["CR"]
        self.cd0_ap = model["CD0"]["AP"]
        self.cd2_ap = model["CD2"]["AP"]
        self.cd0_ld = model["CD0"]["LD"]
        self.cd2_ld = model["CD2"]["LD"]
        self.cd0_lgear = model["CD0_lgear"]

    @ndarrayconvert(column=True)
    def _cd(self, cl, cd0, cd2, cd0_lg=0.0):
        """Compute drag coefficient (BADA3 eq. 3.6-2, 3.6-3, 3.6-4).

        Args:
            cl: Lift coefficient.
            cd0: Parasitic drag coefficient.
            cd2: Induced drag coefficient.
            cd0_lg: Parasitic drag coefficient of landing gear. Defaults to 0.

        Returns:
            Drag coefficient.

        """
        cd = cd0 + cd0_lg + cd2 * cl**2
        return cd

    def clean_drag_polar_params(self, tas=None, alt=None, dT=0):
        """Return clean-configuration drag polar parameters.

        BADA3 clean drag is CD = cd0 + cd2 * CL**2. The optional
        arguments are accepted for API symmetry with Mach-dependent models.
        """
        return {"cd0": self.cd0_cr, "cd2": self.cd2_cr, "cd6": 0.0}

    @ndarrayconvert(column=True)
    def _cl(self, mass, tas, alt, dT=0):
        """Compute lift coefficient (BADA3 eq. 3.6-1).

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Tuple of (lift coefficient, q * S).

        """
        v = tas * self.aero.kts
        h = alt * self.aero.ft
        rho = self.aero.density(h, dT=dT)

        # calculate total lift
        qS = 0.5 * rho * v**2 * self.S
        mass = self.mass_to_kg(mass)
        L = mass * self.aero.g0
        if self.smooth:
            qS_safe = self.backend.smooth_max(qS, 1e-3, softness=1e-4)
        else:
            qS_safe = self.backend.maximum(qS, 1e-3)
        cl = L / qS_safe  # avoid zero division

        return cl, qS

    @ndarrayconvert(column=True)
    def clean(self, mass, tas, alt, vs=None, dT=0) -> float | ndarray:
        """Compute drag in clean configuration (BADA3 eq. 3.6-5).

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            vs: Unused.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Total drag in clean configuration (N).

        """
        cl, qS = self._cl(mass, tas, alt, dT=dT)
        cd = self._cd(cl, self.cd0_cr, self.cd2_cr)
        D = cd * qS
        return D

    @ndarrayconvert(column=True)
    def nonclean(self, mass, tas, alt, landing_gear=False, phase=None, dT=0):
        """Compute drag in non-clean configuration (BADA3 eq. 3.6-3, 3.6-4, 3.6-5).

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            landing_gear: Whether landing gear is extended. Defaults to False.
            phase: Flight phase, one of 'AP' or 'LD'.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Total drag in non-clean configuration (N).

        """
        # ensure that the phase is one of AP or LD
        if phase not in ("AP", "LD"):
            raise ValueError(f"Phase {phase} unknown or not applicable.")
        cl, qS = self._cl(mass, tas, alt, dT=dT)
        if phase == "AP":
            # if both cd0 and cd2 of approach are zero, use clean config
            if self.cd0_ap == 0.0 and self.cd2_ap == 0.0:
                cd = self._cd(cl, self.cd0_cr, self.cd2_cr)
            else:
                cd = self._cd(cl, self.cd0_ap, self.cd2_ap)
        else:  # landing phase
            if self.cd0_ld == 0.0 and self.cd2_ld == 0.0:
                cd = self._cd(cl, self.cd0_cr, self.cd2_cr)
            else:
                cd = self._cd(
                    cl,
                    self.cd0_ld,
                    self.cd2_ld,
                    self.cd0_lgear if landing_gear else 0.0,
                )
        D = cd * qS
        return D


class Thrust(base.ThrustBase):
    """Compute the thrust of an aircraft using BADA3 models."""

    def __init__(self, ac: str, bada_path: str, model=None, **kwargs):
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()
        self.smooth = kwargs.get("smooth", False)
        if not model:
            model = load_bada3(ac, bada_path)
        self.engine_type = model["engine"]["type"]
        self.ct = model["Ct"]
        self.ctdeshigh = model["CTdeshigh"]
        self.ctdeslow = model["CTdeslow"]
        self.ctdesapp = model["CTdesapp"]
        self.ctdesld = model["CTdesld"]

        nc_vals = [
            model["CD0"]["AP"],
            model["CD0"]["LD"],
            model["CD0_lgear"],
            model["CD2"]["AP"],
            model["CD2"]["LD"],
        ]
        nc_avail = not all(val == 0 for val in nc_vals)
        if nc_avail:
            self.hpdes = self.backend.maximum(model["HpDes"], 8000.0)
        else:
            self.hpdes = model["HpDes"]

    @ndarrayconvert(column=True)
    def climb(self, tas, alt, dT=0) -> float | ndarray:
        """Compute maximum climb thrust (BADA3 eq. 3.7-1 to 3.7-7).

        Args:
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Thrust force (N) during the climb phase.

        """

        # calculate maximum climb thrust at ISA
        if self.engine_type == "turbofan":
            thr_isa = self.ct[0] * (1 - alt / self.ct[1] + self.ct[2] * alt**2)
        elif self.engine_type == "turboprop":
            thr_isa = self.ct[0] / tas * (1 - alt / self.ct[1]) + self.ct[2]
        elif self.engine_type in ("piston", "electric"):
            thr_isa = self.ct[0] * (1 - alt / self.ct[1]) + self.ct[2] / tas
        else:
            raise ValueError("Unknown engine type")

        # correct for temperature deviations from ISA whilst considering limits
        # eq. (3.7.4 - 3.7.7)
        dT_eff = dT - self.ct[3]
        b = self.backend
        if self.smooth:
            c_tc5 = b.smooth_max(self.ct[4], 0.0, softness=1e-4)
            dT_lim = b.smooth_clip(c_tc5 * dT_eff, 0.0, 0.4, softness=1e-3)
        else:
            c_tc5 = b.maximum(self.ct[4], 0.0)
            dT_lim = b.maximum(0.0, b.minimum(c_tc5 * dT_eff, 0.4))
        thr_mcl = thr_isa * (1 - dT_lim)
        return thr_mcl

    @ndarrayconvert(column=True)
    def cruise(self, tas, alt, roc=None, dT=0) -> float | ndarray:
        """Compute maximum cruise thrust (BADA3 eq. 3.7-8).

        Args:
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            roc: Unused.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Thrust force (N) during the cruise phase.

        """
        ct_cr = 0.95  # constant in BADA3
        thr_cl_max = self.climb(tas, alt, dT)
        thr_cr_max = ct_cr * thr_cl_max
        return thr_cr_max

    @ndarrayconvert(column=True)
    def takeoff(self, tas, alt, dT=0) -> float | ndarray:
        """Compute takeoff thrust (equal to maximum climb thrust in BADA3).

        Args:
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Thrust force (N) during takeoff.

        """
        return self.climb(tas, alt, dT)

    @ndarrayconvert(column=True)
    def idle(self, tas, alt, roc=None, dT=0, config=None) -> float | ndarray:
        """Compute idle (descent) thrust (BADA3 eq. 3.7-9 to 3.7-12).

        Args:
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            roc: Unused.
            dT: ISA temperature deviation (K). Defaults to 0.
            config: Aircraft configuration ('CR', 'AP', or 'LD').

        Returns:
            Thrust force (N) during idle/descent.

        """
        thr_cl_max = self.climb(tas, alt, dT)
        if config not in ("CR", "AP", "LD"):
            raise ValueError("Config unknown.")
        if config == "CR":
            ctdes = self.ctdeslow
        elif config == "AP":
            ctdes = self.ctdesapp
        else:
            ctdes = self.ctdesld
        high = self.ctdeshigh * thr_cl_max
        low = ctdes * thr_cl_max
        if self.smooth:
            thr_des = self.backend.smooth_switch(
                alt,
                self.hpdes,
                low,
                high,
                softness=100.0,
            )
        else:
            thr_des = self.backend.where(alt > self.hpdes, high, low)
        return thr_des


class FuelFlow(base.FuelFlowBase):
    """Compute the fuel flow of an aircraft using BADA3 models."""

    def __init__(self, ac: str, bada_path: Optional[str] = None, model=None, **kwargs):
        super().__init__(ac, **kwargs)
        self.ac = ac.upper()
        self.smooth = kwargs.get("smooth", False)
        if not model:
            model = load_bada3(ac, bada_path)
        self.thrust = Thrust(
            ac,
            bada_path,
            model=model,
            backend=self.backend,
            smooth=self.smooth,
        )
        self.drag = Drag(
            ac,
            bada_path,
            model=model,
            backend=self.backend,
            smooth=self.smooth,
        )
        # load parameters from BADA3
        self.engine_type = model["engine"]["type"]
        self.cf1 = model["Cf"][0]
        self.cf2 = model["Cf"][1]
        self.cf3 = model["CfDes"][0]
        self.cf4 = model["CfDes"][1]
        self.cfcr = model["CfCrz"]

    @ndarrayconvert(column=True)
    def nominal(self, mass, tas, alt, vs=0, dT=0) -> float | ndarray:
        """Calculate nominal fuel flow (BADA3 eq. 3.9-1, 3.9-2, 3.9-3).

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            vs: Vertical rate (ft/min). Defaults to 0.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Nominal fuel flow (kg/s).

        """
        mass = self.mass_to_kg(mass)
        # calculate gamma angle and drag
        v = tas * self.aero.kts
        gamma = self.backend.arctan2(vs * self.aero.fpm, v)
        D = self.drag.clean(mass, tas, alt, vs, dT=dT)

        # thrust is equal to drag, but not larger than climb thrust
        required_thrust = D + mass * self.aero.g0 * self.backend.sin(gamma)
        climb_thrust = self.thrust.climb(tas, alt, dT=dT)
        if self.smooth:
            T = self.backend.smooth_min(
                required_thrust,
                climb_thrust,
                softness=500.0,
            )
        else:
            T = self.backend.minimum(required_thrust, climb_thrust)

        # calculate nominal fuel flow depending on engine_type
        if self.engine_type == "turbofan":
            eta = self.cf1 * (1 + tas / self.cf2) * 1e-3
            f_nom = eta * T
        elif self.engine_type == "turboprop":
            eta = self.cf1 * (1 - tas / self.cf2) * (tas / 1e3) * 1e-3
            f_nom = eta * T
        elif self.engine_type in ("piston", "electric"):
            f_nom = self.cf1
        else:
            raise ValueError("Unknown engine type.")

        # Compare in kg/min: idle() returns kg/s, while f_nom is kg/min.
        f_idle = self.idle(mass, tas, alt, vs) * 60.0
        if self.smooth:
            # Avoid a near-kink in the fuel dynamics. This width bounds
            # the idle-floor smoothing bias by 0.0025 kg/s (0.3 / 120).
            f_nom = self.backend.smooth_max(f_nom, f_idle, softness=0.3)
        else:
            f_nom = self.backend.maximum(f_nom, f_idle)
        return f_nom / 60.0  # conversion [kg/min] -> [kg/s]

    @ndarrayconvert(column=True)
    def enroute(self, mass, tas, alt, vs=0, acc=None, dT=0):
        """Calculate nominal enroute fuel flow.

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            vs: Vertical rate (ft/min). Defaults to 0.
            acc: Unused.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Nominal fuel flow (kg/s).

        """
        f_nom = self.nominal(mass, tas, alt, vs, dT=dT)  # [kg/s]
        return f_nom

    @ndarrayconvert(column=True)
    def cruise(self, mass, tas, alt, vs=0, acc=None, dT=0):
        """Calculate cruise fuel flow with the BADA3 cruise correction (eq. 3.9-6).

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            vs: Vertical rate (ft/min). Defaults to 0.
            acc: Unused.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Cruise fuel flow (kg/s).

        """
        f_nom = self.nominal(mass, tas, alt, vs, dT=dT)  # [kg/s]
        return self.cfcr * f_nom

    @ndarrayconvert(column=True)
    def idle(self, mass, tas, alt, vs=None):
        """Calculate idle fuel flow (BADA3 eq. 3.9-4).

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: Unused.
            alt: Geopotential pressure altitude (ft).
            vs: Unused.

        Returns:
            Idle fuel flow (kg/s).

        """
        self.mass_to_kg(mass)
        if self.engine_type in ("turbofan", "turboprop"):
            f_min = self.cf3 * (1 - alt / self.cf4)
        elif self.engine_type in ("piston", "electric"):
            f_min = self.cf3
        else:
            raise ValueError("Unknown engine type.")
        return f_min / 60.0  # conversion [kg/min] -> [kg/s]

    @ndarrayconvert(column=True)
    def approach(self, mass, tas, alt, vs=0, dT=0):
        """Calculate approach/landing fuel flow (BADA3 eq. 3.9-5).

        Only applicable to jet and turboprop aircraft.

        Args:
            mass: Aircraft mass in kg or an MTOW fraction in (0, 1].
            tas: True airspeed (kt).
            alt: Geopotential pressure altitude (ft).
            vs: Vertical rate (ft/min). Defaults to 0.
            dT: ISA temperature deviation (K). Defaults to 0.

        Returns:
            Approach/landing fuel flow (kg/s).

        """
        if self.engine_type not in ("turbofan", "turboprop"):
            raise ValueError(
                f"Engine type {self.engine_type} unknown or not applicable."
            )
        f_nom = self.nominal(mass, tas, alt, vs, dT=dT)  # [kg/s]
        f_min = self.idle(mass, tas, alt, vs)  # [kg/s]
        if self.smooth:
            f_ap = self.backend.smooth_max(f_nom, f_min, softness=1e-4)
        else:
            f_ap = self.backend.maximum(f_nom, f_min)
        return f_ap
