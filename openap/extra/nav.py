"""Navigation module for accessing navigation databases."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openap.extra import aero

fixes = None
airports = None

curr_path = os.path.dirname(os.path.realpath(__file__))
db_airport = curr_path + "/../data/nav/airports.csv"
db_fix = curr_path + "/../data/nav/fix.dat"


def _read_fix():
    return pd.read_csv(
        db_fix,
        skiprows=3,
        skipfooter=1,
        engine="python",
        sep=r"\s+",
        names=("lat", "lon", "fix"),
        encoding="unicode_escape",
    )


def _read_airport():
    return pd.read_csv(db_airport)


def airport(name: str) -> Optional[Dict]:
    """Get the airport information.

    Args:
        name: ICAO code of the airport.

    Returns:
        Information related to the airport, including position,
        country, and region information. None if not found.

    """
    NAME = str(name).upper()

    if not isinstance(airport, pd.DataFrame):
        airports = _read_airport()

    df = airports[airports["icao"] == NAME]
    if df.shape[0] == 0:
        return None
    else:
        return df.iloc[0, :].to_dict()


def closest_airport(lat: float, lon: float) -> Optional[str]:
    """Get the closest airport to a location.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        ICAO code of the airport, or None if not found.

    """
    global airports

    if not isinstance(airport, pd.DataFrame):
        airports = _read_airport()

    df = airports[
        airports["lat"].between(lat - 2, lat + 2)
        & airports["lon"].between(lon - 2, lon + 2)
    ]

    if df.shape[0] == 0:
        return None

    coords = np.array(df[["lat", "lon"]])
    dist2 = np.sum((coords - [lat, lon]) ** 2, axis=1)
    idx = np.argmin(dist2)

    ap = df.iloc[idx, :]

    return ap.icao


def fix(name: str) -> List:
    """Get position of a fix or waypoint.

    Args:
        name: Name of the fix or waypoint.

    Returns:
        Latitude and longitude.

    """
    global fixes

    if not isinstance(fixes, pd.DataFrame):
        fixes = _read_fix()

    NAME = str(name).upper()
    fix = fixes[fixes["fix"] == NAME].iloc[0].tolist()
    return fix


def closest_fix(lat: float, lon: float) -> Tuple[List, int]:
    """Get the closest fix to a location.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        Tuple of (fix coordinates, distance to the fix in meters).

    """
    global fixes

    if not isinstance(fixes, pd.DataFrame):
        fixes = _read_fix()

    mask = (fixes["lat"].between(lat - 1, lat + 1)) & (
        fixes["lon"].between(lon - 1, lon + 1)
    )
    chunk = fixes[mask]

    lats = np.asarray(chunk["lat"])
    lons = np.asarray(chunk["lon"])

    distances = aero.distance(lat, lon, lats, lons)
    idx = distances.argmin()

    fix = chunk.iloc[idx].tolist()
    dist = distances[idx]

    return fix, int(dist)
