from typing import Union

import numpy as np

from . import prop


def from_range(
    typecode: str,
    distance: float,
    load_factor: float = 0.8,
    fraction: bool = False,
    **kwargs,
) -> float:
    """Compute aircraft mass based on range, load factor, and fraction settings.

    This function calculates the aircraft mass considering fuel and payload weights
    based on the given flight distance and load factor.

    Args:
        typecode: ICAO aircraft type code (e.g. A320, B738).
        distance: Flight distance in nautical miles.
        load_factor: Load factor between 0 and 1. Defaults to 0.8.
        fraction: If True, return mass fraction of MTOW. Defaults to False.

    Returns:
        Aircraft mass in kg, or mass fraction if fraction=True.

    """
    ac = prop.aircraft(typecode, **kwargs)

    range_fraction = distance / ac["cruise"]["range"]
    range_fraction = np.clip(range_fraction, 0.2, 1)

    max_fuel_weight = ac["mfc"] * 0.8025  # L->kg
    fuel_weight = range_fraction * max_fuel_weight

    payload_weight = (ac["mtow"] - max_fuel_weight - ac["oew"]) * load_factor

    mass = ac["oew"] + fuel_weight + payload_weight

    if fraction:
        return mass / ac["mtow"]
    else:
        return mass
