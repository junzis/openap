"""Retrieve properties of aircraft and engines."""

import glob
import logging
import os
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

warnings.simplefilter("once", UserWarning)

curr_path = os.path.dirname(os.path.realpath(__file__))
dir_aircraft = os.path.join(curr_path, "data/aircraft/")
file_engine = os.path.join(curr_path, "data/engine/engines.csv")
file_synonym = os.path.join(curr_path, "data/aircraft/_synonym.csv")

aircraft_synonym = pd.read_csv(file_synonym)


@lru_cache()
def available_aircraft(use_synonym: bool = False) -> List[str]:
    """Get available aircraft types in OpenAP model.

    Args:
        use_synonym: Include synonyms in the list. Defaults to False.

    Returns:
        List of aircraft type codes.

    """
    files = sorted(glob.glob(dir_aircraft + "*.yml"))
    acs = [f[-8:-4] for f in files]

    if use_synonym:
        syno = aircraft_synonym.orig.to_list()
        acs = acs + syno

    return acs


def aircraft(ac: str, use_synonym: bool = False, **kwargs) -> Dict[str, Any]:
    """Get details of an aircraft type.

    Args:
        ac: ICAO aircraft type (for example: A320).
        use_synonym: Use synonym if aircraft not found. Defaults to False.

    Returns:
        Performance parameters related to the aircraft.

    """
    ac = ac.lower()

    files = glob.glob(dir_aircraft + ac + ".yml")

    if len(files) == 0:
        syno = aircraft_synonym.query("orig==@ac")
        if use_synonym and syno.shape[0] > 0:
            new_ac = syno.new.iloc[0]
            files = glob.glob(dir_aircraft + new_ac + ".yml")
            warnings.warn(
                f"Aircraft: using synonym {new_ac} for {ac}",
                UserWarning,
                stacklevel=0,
            )
        elif use_synonym:
            raise ValueError(f"Aircraft {ac} not available, and no synonym found.")
        else:
            raise ValueError(
                f"Aircraft {ac} not available. "
                "Try to set `use_synonym=True` to initialize the object."
            )

    f = files[0]
    with open(f, "r") as file:
        acdict = yaml.safe_load(file.read())

    # compatibility with old aircraft files
    acdict["limits"] = dict(
        MTOW=acdict.get("mtow"),
        MLW=acdict.get("mlw"),
        OEW=acdict.get("oew"),
        MFC=acdict.get("mfc"),
        VMO=acdict.get("vmo"),
        MMO=acdict.get("mmo"),
        ceiling=acdict.get("ceiling"),
    )

    return acdict


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
