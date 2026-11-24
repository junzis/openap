"""Retrieve properties of aircraft and engines.

The aircraft properties come from the Aerox aircraft records
(https://github.com/open-aviation/aerox).
"""

import copy
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import aerox

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

curr_path = os.path.dirname(os.path.realpath(__file__))
file_engine = os.path.join(curr_path, "data/engine/engines.csv")

_FT = 0.3048
_NM_TO_KM = 1.852


@lru_cache()
def available_aircraft(use_synonym: bool = False) -> List[str]:
    """Get available aircraft types in OpenAP model.

    These are the turbofan aircraft in Aerox whose default engine is in the
    OpenAP engine table, so that all OpenAP models can be built for them.

    Args:
        use_synonym: Has no effect. Kept for compatibility.

    Returns:
        List of aircraft type codes, in lower case.

    """
    engine_names = set(pd.read_csv(file_engine).name.str.upper())
    codes = []
    for code in aerox.get_available_aircraft():
        record_engine = aerox.load_aircraft(code).engine
        if record_engine is None or record_engine.type != "turbofan":
            continue
        default = (record_engine.default or "").strip().upper()
        if any(name.startswith(default) for name in engine_names) and default:
            codes.append(code.lower())
    return codes


@lru_cache()
def _aircraft(ac: str) -> Dict[str, Any]:
    code = ac.upper()
    if code not in aerox.get_available_aircraft():
        raise ValueError(f"Aircraft {ac} not available.")
    record = aerox.load_aircraft(code)

    def group(obj, *names, rename=None):
        if obj is None:
            return {name: None for name in names}
        rename = rename or {}
        return {rename.get(name, name): getattr(obj, name) for name in names}

    kinematic = record.kinematic
    cruise_range_nm = getattr(kinematic, "cruise_range_nm", None)
    engine = group(record.engine, "type", "mount", "number", "default")
    engine["options"] = list(record.engine.options or []) if record.engine else []
    acdict = {
        "aircraft": record.aircraft,
        "mtow": record.mtow,
        "mlw": record.mlw,
        "oew": record.oew,
        "mfc": record.mfc,  # litres
        "vmo": record.vmo,  # kt
        "mmo": record.mmo,
        "ceiling": None if record.ceiling is None else record.ceiling * _FT,  # m
        "pax": group(record.pax, "max", "low", "high"),
        "fuselage": group(record.fuselage, "length", "height", "width"),
        "wing": group(
            record.wing, "area", "span", "mac", "sweep", "t_c", rename={"t_c": "t/c"}
        ),
        "flaps": group(
            record.flaps,
            "type",
            "area",
            "bf_b",
            "lambda_f",
            "cf_c",
            "Sf_S",
            rename={"bf_b": "bf/b", "cf_c": "cf/c", "Sf_S": "Sf/S"},
        ),
        "cruise": {
            "mach": getattr(kinematic, "cruise_mach", None),
            "range": None if cruise_range_nm is None else cruise_range_nm * _NM_TO_KM,
        },
        "engine": engine,
    }
    acdict["limits"] = dict(
        MTOW=acdict["mtow"],
        MLW=acdict["mlw"],
        OEW=acdict["oew"],
        MFC=acdict["mfc"],
        VMO=acdict["vmo"],
        MMO=acdict["mmo"],
        ceiling=acdict["ceiling"],
    )
    return acdict


def aircraft(ac: str, use_synonym: bool = False, **kwargs) -> Dict[str, Any]:
    """Get details of an aircraft type.

    The values come from the Aerox aircraft record. Units: masses in kg,
    ``mfc`` in litres, ``vmo`` in kt, ``ceiling`` in m, lengths in m, areas in
    m², ``cruise.range`` in km.

    Args:
        ac: ICAO aircraft type (for example: A320).
        use_synonym: Has no effect. Kept for compatibility.

    Returns:
        Performance parameters related to the aircraft.

    """
    return copy.deepcopy(_aircraft(ac.upper()))


@lru_cache()
def aircraft_engine_options(ac: str) -> List[str]:
    """Get engine options of an aircraft type.

    Args:
        ac: ICAO aircraft type (for example: A320).

    Returns:
        Engine options for the aircraft.

    """
    acdict = aircraft(ac)

    if isinstance(acdict["engine"]["options"], dict):
        eng_options = list(acdict["engine"]["options"].values())
    elif isinstance(acdict["engine"]["options"], list):
        eng_options = list(acdict["engine"]["options"])

    return eng_options


@lru_cache()
def search_engine(eng: str) -> Optional[List[str]]:
    """Search engine by the starting characters.

    Args:
        eng: Engine type (for example: CFM56-5).

    Returns:
        Matching engine types, or None if not found.

    """
    ENG = eng.strip().upper()  # noqa: F841 - used in query via @ENG
    engines = pd.read_csv(file_engine)

    available_engines = engines.query("name.str.startswith(@ENG)", engine="python")

    if available_engines.shape[0] == 0:
        logger.info("Engine not found.")
        result = None
    else:
        result = available_engines.name.tolist()
        logger.info("Engines found: %s", result)

    return result


@lru_cache()
def engine(eng: str) -> Dict[str, Any]:
    """Get engine parameters.

    Args:
        eng: Engine type (for example: CFM56-5B6).

    Returns:
        Engine parameters.

    """
    ENG = eng.strip().upper()  # noqa: F841 - used in query via @ENG
    engines = pd.read_csv(file_engine)

    # try to look for the unique engine
    available_engines = engines.query(
        "name.str.upper().str.startswith(@ENG)", engine="python"
    )
    if available_engines.shape[0] >= 1:
        available_engines.index = available_engines.name

        seleng = available_engines.to_dict(orient="records")[0]
        seleng["name"] = eng

        # compute fuel flow correction factor kg/s/N per meter
        if np.isfinite(seleng["cruise_sfc"]):
            sfc_cr = seleng["cruise_sfc"]
            sfc_to = seleng["ff_to"] / (seleng["max_thrust"] / 1000)
            fuel_ch = np.round((sfc_cr - sfc_to) / (seleng["cruise_alt"] * 0.3048), 8)
        else:
            # see openap paper
            fuel_ch = 6.7e-7

        seleng["fuel_ch"] = fuel_ch
    else:
        raise ValueError(f"Data for engine {eng} not found.")

    return seleng
