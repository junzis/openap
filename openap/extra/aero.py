"""Aero library - re-exports from openap.aero for backward compatibility.

Functions for aeronautics in this module
    - physical quantities always in SI units
    - lat,lon,course and heading in degrees

International Standard Atmosphere
    p,rho,T = atmos(h)    # atmos as function of geopotential altitude h [m]
    a = vsound(h)         # speed of sound [m/s] as function of h[m]
    p = pressure(h)       # calls atmos but returns only pressure [Pa]
    T = temperature(h)    # calculates temperature [K]
    rho = density(h)      # calls atmos but returns only density [kg/m3]

Speed conversion at altitude h[m] in ISA:
    mach = tas2mach(v_tas,h)    # true airspeed (v_tas) to mach number conversion
    v_tas = mach2tas(mach,h)    # true airspeed (v_tas) to mach number conversion
    v_tas = eas2tas(v_eas,h)    # equivalent airspeed to true airspeed, h in [m]
    v_eas = tas2eas(v_tas,h)    # true airspeed to equivalent airspeed, h in [m]
    v_tas = cas2tas(v_cas,h)    # v_cas  to v_tas conversion both m/s, h in [m]
    v_cas = tas2cas(v_tas,h)    # v_tas to v_cas conversion both m/s, h in [m]
    v_cas = mach2cas(mach,h)    # mach to v_cas conversion v_cas in m/s, h in [m]
    mach   = cas2mach(v_cas,h)  # v_cas to mach conversion v_cas in m/s, h in [m]
"""

# Re-export everything from the main aero module
from openap.aero import (
    T0,
    R,
    # Constants
    a0,
    # Functions
    atmos,
    bearing,
    beta,
    cas2mach,
    cas2tas,
    crossover_alt,
    density,
    distance,
    eas2tas,
    fpm,
    ft,
    g0,
    gamma,
    gamma1,
    gamma2,
    h_isa,
    inch,
    kts,
    latlon,
    lbs,
    mach2cas,
    mach2tas,
    nm,
    p0,
    pressure,
    r_earth,
    rho0,
    sqft,
    tas2cas,
    tas2eas,
    tas2mach,
    temperature,
    vsound,
)

__all__ = [
    # Constants
    "kts",
    "ft",
    "fpm",
    "inch",
    "sqft",
    "nm",
    "lbs",
    "g0",
    "R",
    "p0",
    "rho0",
    "T0",
    "gamma",
    "gamma1",
    "gamma2",
    "beta",
    "r_earth",
    "a0",
    # Functions
    "atmos",
    "temperature",
    "pressure",
    "density",
    "vsound",
    "distance",
    "bearing",
    "h_isa",
    "latlon",
    "tas2mach",
    "mach2tas",
    "eas2tas",
    "tas2eas",
    "cas2tas",
    "tas2cas",
    "mach2cas",
    "cas2mach",
    "crossover_alt",
]
