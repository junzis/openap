"""aero.py adapted for CasADi"""

from casadi import casadi

from . import numpy_override as np

"""Aero and Geo Constants """
kts = 0.514444  # knot -> m/s
ft = 0.3048  # ft -> m
fpm = 0.00508  # ft/min -> m/s
inch = 0.0254  # inch -> m
sqft = 0.09290304  # 1 square foot
nm = 1852.0  # nautical mile -> m
lbs = 0.453592  # pound -> kg
g0 = 9.80665  # m/s2, Sea level gravity constant
R = 287.05287  # m2/(s2 x K), gas constant, sea level ISA
p0 = 101325.0  # Pa, air pressure, sea level ISA
rho0 = 1.225  # kg/m3, air density, sea level ISA
T0 = 288.15  # K, temperature, sea level ISA
gamma = 1.40  # cp/cv for air
gamma1 = 0.2  # (gamma-1)/2 for air
gamma2 = 3.5  # gamma/(gamma-1) for air
beta = -0.0065  # [K/m] ISA temp gradient below tropopause
r_earth = 6371000.0  # m, average earth radius
a0 = 340.293988  # m/s, sea level speed of sound ISA, sqrt(gamma*R*T0)

deg = 180 / 3.14159  # radians -> degrees
rad = 3.14159 / 180  # degrees -> radians


def atmos(h, dT=0):
    """Compute press, density and temperature at a given altitude.

    Args:
        h (SX or MX): Altitude (in meters).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        (SX, SX, SX) or (MX, MX, MX):
            Air pressure (Pa), density (kg/m3), and temperature (K).

    """
    assert -25 < dT < 15

    T0_ = T0 + dT
    T = 0.65 * np.log(1 + np.exp(-10 * (h / 1000 - 11))) + 216.65
    rho = rho0 * ((T0_ + beta * h) / T0_) ** 4.2559
    p = rho * R * T
    return p, rho, T


def temperature(h, dT=0):
    """Compute air temperature at a given altitude.

    Args:
        h (SX or MX): Altitude (in meters).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: Air temperature (K).

    """
    p, r, T = atmos(h, dT)
    return T


def pressure(h, dT=0):
    """Compute air pressure at a given altitude.

    Args:
        h (SX or MX): Altitude (in meters).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: Air pressure (Pa).

    """
    p, r, T = atmos(h, dT)
    return p


def density(h, dT=0):
    """Compute air density at a given altitude.

    Args:
        h (SX or MX): Altitude (in meters).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: Air density (kg/m3).

    """
    p, r, T = atmos(h, dT)
    return r


def vsound(h, dT=0):
    """Compute speed of sound at a given altitude.

    Args:
        h (SX or MX): Altitude (in meters).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: speed of sound (m/s).

    """
    T = temperature(h, dT)
    a = np.sqrt(gamma * R * T)
    return a


def distance(lat1, lon1, lat2, lon2, h=0):
    """Compute distance between two (or two series) of coordinates using Harversine formula.

    Args:
        lat1 (SX or MX): Starting latitude (in degrees).
        lon1 (SX or MX): Starting longitude (in degrees).
        lat2 (SX or MX): Ending latitude (in degrees).
        lon2 (SX or MX): Ending longitude (in degrees).
        h (SX or MX): Altitude (in meters). Defaults to 0.

    Returns:
        SX or MX: Distance (in meters).

    """
    # convert decimal degrees to radians
    lat1 = lat1 * rad
    lon1 = lon1 * rad
    lat2 = lat2 * rad
    lon2 = lon2 * rad

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = c * (r_earth + h)  # meters, radius of earth
    return dist


def bearing(lat1, lon1, lat2, lon2):
    """Compute the bearing between two (or two series) of coordinates.

    Args:
        lat1 (SX or MX): Starting latitude (in degrees).
        lon1 (SX or MX): Starting longitude (in degrees).
        lat2 (SX or MX): Ending latitude (in degrees).
        lon2 (SX or MX): Ending longitude (in degrees).

    Returns:
        SX or MX: Bearing (in degrees). Between 0 and 360.

    """
    lat1 = lat1 * rad
    lon1 = lon1 * rad
    lat2 = lat2 * rad
    lon2 = lon2 * rad
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    initial_bearing = np.arctan2(x, y)
    initial_bearing = initial_bearing * deg
    bearing = (initial_bearing + 360) % 360
    return bearing


def h_isa(p, dT=0):
    """Compute ISA altitude for a given pressure.

    Args:
        p (SX or MX): Pressure (in Pa).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: altitude (m).

    """
    # p >= 22630:
    T0_shift = T0 + dT
    T = T0_shift * (p0 / p) ** ((-0.0065 * R) / g0)
    h = (T - T0_shift) / -0.0065

    # 5470 < p < 22630
    T1 = T0_shift - 0.0065 * (11000)
    p1 = 22630
    h1 = -R * T1 / g0 * np.log(p / p1) + 11000

    h_ = casadi.if_else(p > 22630, h, h1)

    return h_


def latlon(lat1, lon1, d, brg, h=0):
    """Get lat/lon given current point, distance and bearing.

    Args:
        lat1 (SX or MX): Starting latitude (in degrees).
        lon1 (SX or MX): Starting longitude (in degrees).
        d (SX or MX): distance from point 1 (meters)
        brg (SX or MX): bearing at point 1 (in degrees)
        h (SX or MX): Altitude (in meters). Defaults to 0.

    Returns:
        lat2: Point latitude.
        lon2: Point longitude

    """
    # convert decimal degrees to radians
    lat1 = lat1 * rad
    lon1 = lon1 * rad
    brg = brg * rad

    # haversine formula
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d / (r_earth + h))
        + np.cos(lat1) * np.sin(d / (r_earth + h)) * np.cos(brg)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(brg) * np.sin(d / (r_earth + h)) * np.cos(lat1),
        np.cos(d / (r_earth + h)) - np.sin(lat1) * np.sin(lat2),
    )
    lat2 = lat2 * deg
    lon2 = lon2 * deg
    return lat2, lon2


def tas2mach(v_tas, h, dT=0):
    """Convert true airspeed to mach number at a given altitude.

    Args:
        v_tas (SX or MX): True airspeed (m/s).
        h (SX or MX): Altitude (m).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: mach number.

    """
    a = vsound(h, dT)
    mach = v_tas / a
    return mach


def mach2tas(mach, h, dT=0):
    """Convert mach number to true airspeed at a given altitude.

    Args:
        mach (SX or MX): Mach number.
        h (SX or MX): Altitude (m).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: True airspeed (m/s).

    """
    a = vsound(h, dT)
    v_tas = mach * a
    return v_tas


def eas2tas(v_eas, h, dT=0):
    """Convert equivalent airspeed to true airspeed at a given altitude.

    Args:
        v_eas (SX or MX): Equivalent airspeed (m/s).
        h (SX or MX): Altitude (m).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: True airspeed (m/s).

    """
    rho = density(h, dT)
    v_tas = v_eas * np.sqrt(rho0 / rho)
    return v_tas


def tas2eas(v_tas, h, dT=0):
    """Convert true airspeed to equivalent airspeed at a given altitude.

    Args:
        v_tas (SX or MX): True airspeed (m/s).
        h (SX or MX): Altitude (m).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: Equivalent airspeed (m/s).

    """
    rho = density(h, dT)
    v_eas = v_tas * np.sqrt(rho / rho0)
    return v_eas


def cas2tas(v_cas, h, dT=0):
    """Convert calibrated airspeed to true airspeed at a given altitude.

    Args:
        v_cas (SX or MX): Equivalent airspeed (m/s).
        h (SX or MX): Altitude (m).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: True airspeed (m/s).

    """
    p, rho, T = atmos(h, dT)
    qdyn = p0 * ((1.0 + rho0 * v_cas * v_cas / (7.0 * p0)) ** 3.5 - 1.0)
    v_tas = np.sqrt(7.0 * p / rho * ((1.0 + qdyn / p) ** (2.0 / 7.0) - 1.0))
    return v_tas


def tas2cas(v_tas, h, dT=0):
    """Convert true airspeed to calibrated airspeed at a given altitude.

    Args:
        v_tas (SX or MX): True airspeed (m/s).
        h (SX or MX): Altitude (m).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: Calibrated airspeed (m/s).

    """
    p, rho, T = atmos(h, dT)
    qdyn = p * ((1.0 + rho * v_tas * v_tas / (7.0 * p)) ** 3.5 - 1.0)
    v_cas = np.sqrt(7.0 * p0 / rho0 * ((qdyn / p0 + 1.0) ** (2.0 / 7.0) - 1.0))
    return v_cas


def mach2cas(mach, h, dT=0):
    """Convert mach number to calibrated airspeed at a given altitude.

    Args:
        mach (SX or MX): Mach number.
        h (SX or MX): Altitude (m).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: Calibrated airspeed (m/s).

    """
    v_tas = mach2tas(mach, h, dT)
    v_cas = tas2cas(v_tas, h, dT)
    return v_cas


def cas2mach(v_cas, h, dT=0):
    """Convert calibrated airspeed to mach number  at a given altitude.

    Args:
        v_cas (SX or MX): Calibrated airspeed (m/s).
        h (SX or MX): Altitude (m).
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: Mach number.

    """
    v_tas = cas2tas(v_cas, h, dT)
    mach = tas2mach(v_tas, h, dT)
    return mach


def crossover_alt(v_cas, mach, dT=0):
    """Convert the crossover altitude given constant CAS and Mach.

    Args:
        v_cas (SX or MX): Calibrated airspeed (m/s).
        mach (SX or MX): Mach number.
        dT (SX or MX): Temperature shift from ISA (in K).  Defaults to 0.

    Returns:
        SX or MX: Altitude (m).

    """
    T0_shift = T0 + dT
    mach = 1e-4 if mach < 1e-4 else mach
    delta = ((0.2 * (v_cas / a0) ** 2 + 1) ** 3.5 - 1) / (
        (0.2 * mach**2 + 1) ** 3.5 - 1
    )
    h = T0_shift / beta * (delta ** (-1 * R * beta / g0) - 1)
    return h
