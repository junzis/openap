"""Aeronautical calculations with backend support.

This module provides atmospheric and airspeed conversion functions
that work with any backend (NumPy, CasADi, JAX).

For backward compatibility, module-level functions use NumPy.
For other backends, use the Aero class directly.

Usage:
    # NumPy (default) - module functions
    from openap import aero
    T = aero.temperature(10000)

    # With specific backend
    from openap.aero import Aero
    from openap.backends import JaxBackend
    aero_jax = Aero(backend=JaxBackend())
    T = aero_jax.temperature(10000)
"""

from typing import Any, Optional, Tuple

from openap.backends import BackendType, NumpyBackend, default_backend

# =============================================================================
# Constants (always available at module level)
# =============================================================================

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


# =============================================================================
# Aero class with backend support
# =============================================================================


class Aero:
    """Aeronautical calculations with configurable backend.

    This class provides all atmospheric and airspeed conversion functions
    using the specified math backend.

    Args:
        backend: Math backend to use. Defaults to NumpyBackend.
    """

    # Constants as class attributes for easy access
    kts = kts
    ft = ft
    fpm = fpm
    inch = inch
    sqft = sqft
    nm = nm
    lbs = lbs
    g0 = g0
    R = R
    p0 = p0
    rho0 = rho0
    T0 = T0
    gamma = gamma
    gamma1 = gamma1
    gamma2 = gamma2
    beta = beta
    r_earth = r_earth
    a0 = a0

    def __init__(self, backend: Optional[BackendType] = None):
        """Initialize Aero with a backend.

        Args:
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        self.backend = backend or default_backend()

    def atmos(self, h: Any, dT: Any = 0) -> Tuple[Any, Any, Any]:
        """Compute pressure, density and temperature at altitude.

        Args:
            h: Altitude (in meters).
            dT: Temperature shift from ISA (in K). Defaults to 0.

        Returns:
            Tuple of (pressure [Pa], density [kg/m3], temperature [K]).
        """
        b = self.backend
        dT = b.clip(dT, -25, 15)
        T0_shift = T0 + dT

        T = b.maximum(T0_shift + beta * h, 216.65 + dT)
        rhotrop = rho0 * b.power(T / T0_shift, 4.256848030018761)
        dhstrat = b.maximum(0.0, h - 11000.0)
        rho = rhotrop * b.exp(-dhstrat / 6341.552161)
        p = rho * R * T
        return p, rho, T

    def temperature(self, h: Any, dT: Any = 0) -> Any:
        """Compute air temperature at altitude.

        Args:
            h: Altitude (in meters).
            dT: Temperature shift from ISA (in K). Defaults to 0.

        Returns:
            Air temperature (K).
        """
        _, _, T = self.atmos(h, dT=dT)
        return T

    def pressure(self, h: Any, dT: Any = 0) -> Any:
        """Compute air pressure at altitude.

        Args:
            h: Altitude (in meters).
            dT: Temperature shift from ISA (in K). Defaults to 0.

        Returns:
            Air pressure (Pa).
        """
        p, _, _ = self.atmos(h, dT=dT)
        return p

    def density(self, h: Any, dT: Any = 0) -> Any:
        """Compute air density at altitude.

        Args:
            h: Altitude (in meters).
            dT: Temperature shift from ISA (in K). Defaults to 0.

        Returns:
            Air density (kg/m3).
        """
        _, rho, _ = self.atmos(h, dT=dT)
        return rho

    def vsound(self, h: Any, dT: Any = 0) -> Any:
        """Compute speed of sound at altitude.

        Args:
            h: Altitude (in meters).
            dT: Temperature shift from ISA (in K). Defaults to 0.

        Returns:
            Speed of sound (m/s).
        """
        T = self.temperature(h, dT=dT)
        return self.backend.sqrt(gamma * R * T)

    def h_isa(self, p: Any, dT: Any = 0) -> Any:
        """Compute ISA altitude for a given pressure.

        Args:
            p: Pressure (Pa).
            dT: Temperature shift from ISA (in K). Defaults to 0.

        Returns:
            Altitude (m).
        """
        b = self.backend
        T0_shift = T0 + dT

        # Troposphere: p >= 22630 Pa
        T = T0_shift * b.power(p0 / p, (beta * R) / g0)
        h = (T - T0_shift) / beta

        # Stratosphere: p < 22630 Pa
        T1 = T0_shift + beta * 11000
        p1 = 22630
        h1 = -R * T1 / g0 * b.log(p / p1) + 11000

        return b.where(p > 22630, h, h1)

    def tas2mach(self, v_tas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert true airspeed to Mach number.

        Args:
            v_tas: True airspeed (m/s).
            h: Altitude (m).
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            Mach number.
        """
        a = self.vsound(h, dT=dT)
        return v_tas / a

    def mach2tas(self, mach: Any, h: Any, dT: Any = 0) -> Any:
        """Convert Mach number to true airspeed.

        Args:
            mach: Mach number.
            h: Altitude (m).
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            True airspeed (m/s).
        """
        a = self.vsound(h, dT=dT)
        return mach * a

    def eas2tas(self, v_eas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert equivalent airspeed to true airspeed.

        Args:
            v_eas: Equivalent airspeed (m/s).
            h: Altitude (m).
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            True airspeed (m/s).
        """
        rho = self.density(h, dT=dT)
        return v_eas * self.backend.sqrt(rho0 / rho)

    def tas2eas(self, v_tas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert true airspeed to equivalent airspeed.

        Args:
            v_tas: True airspeed (m/s).
            h: Altitude (m).
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            Equivalent airspeed (m/s).
        """
        rho = self.density(h, dT=dT)
        return v_tas * self.backend.sqrt(rho / rho0)

    def cas2tas(self, v_cas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert calibrated airspeed to true airspeed.

        Args:
            v_cas: Calibrated airspeed (m/s).
            h: Altitude (m).
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            True airspeed (m/s).
        """
        b = self.backend
        p, rho, _ = self.atmos(h, dT=dT)
        qdyn = p0 * (b.power(1.0 + rho0 * v_cas * v_cas / (7.0 * p0), 3.5) - 1.0)
        v_tas = b.sqrt(7.0 * p / rho * (b.power(1.0 + qdyn / p, 2.0 / 7.0) - 1.0))
        return v_tas

    def tas2cas(self, v_tas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert true airspeed to calibrated airspeed.

        Args:
            v_tas: True airspeed (m/s).
            h: Altitude (m).
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            Calibrated airspeed (m/s).
        """
        b = self.backend
        p, rho, _ = self.atmos(h, dT=dT)
        qdyn = p * (b.power(1.0 + rho * v_tas * v_tas / (7.0 * p), 3.5) - 1.0)
        v_cas = b.sqrt(7.0 * p0 / rho0 * (b.power(qdyn / p0 + 1.0, 2.0 / 7.0) - 1.0))
        return v_cas

    def mach2cas(self, mach: Any, h: Any, dT: Any = 0) -> Any:
        """Convert Mach number to calibrated airspeed.

        Args:
            mach: Mach number.
            h: Altitude (m).
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            Calibrated airspeed (m/s).
        """
        v_tas = self.mach2tas(mach, h, dT=dT)
        return self.tas2cas(v_tas, h, dT=dT)

    def cas2mach(self, v_cas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert calibrated airspeed to Mach number.

        Args:
            v_cas: Calibrated airspeed (m/s).
            h: Altitude (m).
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            Mach number.
        """
        v_tas = self.cas2tas(v_cas, h, dT=dT)
        return self.tas2mach(v_tas, h, dT=dT)

    def crossover_alt(self, v_cas: Any, mach: Any, dT: Any = 0) -> Any:
        """Compute crossover altitude for constant CAS and Mach.

        Args:
            v_cas: Calibrated airspeed (m/s).
            mach: Mach number.
            dT: Temperature shift from ISA (K). Defaults to 0.

        Returns:
            Crossover altitude (m).
        """
        b = self.backend
        T0_shift = T0 + dT
        mach_safe = b.maximum(mach, 1e-4)
        delta = (b.power(0.2 * (v_cas / a0) ** 2 + 1, 3.5) - 1) / (
            b.power(0.2 * mach_safe**2 + 1, 3.5) - 1
        )
        h = T0_shift / beta * (b.power(delta, -1 * R * beta / g0) - 1)
        return h


# =============================================================================
# Module-level functions for backward compatibility (using NumPy)
# =============================================================================

_default_aero = Aero(backend=NumpyBackend())


def atmos(h, dT=0):
    """Compute pressure, density and temperature at altitude."""
    return _default_aero.atmos(h, dT)


def temperature(h, dT=0):
    """Compute air temperature at altitude."""
    return _default_aero.temperature(h, dT)


def pressure(h, dT=0):
    """Compute air pressure at altitude."""
    return _default_aero.pressure(h, dT)


def density(h, dT=0):
    """Compute air density at altitude."""
    return _default_aero.density(h, dT)


def vsound(h, dT=0):
    """Compute speed of sound at altitude."""
    return _default_aero.vsound(h, dT)


def h_isa(p, dT=0):
    """Compute ISA altitude for a given pressure."""
    return _default_aero.h_isa(p, dT)


# =============================================================================
# Backward compatibility - geographic functions moved to openap.geo
# =============================================================================

from openap.geo import bearing, distance, latlon


def tas2mach(v_tas, h, dT=0):
    """Convert true airspeed to Mach number."""
    return _default_aero.tas2mach(v_tas, h, dT)


def mach2tas(mach, h, dT=0):
    """Convert Mach number to true airspeed."""
    return _default_aero.mach2tas(mach, h, dT)


def eas2tas(v_eas, h, dT=0):
    """Convert equivalent airspeed to true airspeed."""
    return _default_aero.eas2tas(v_eas, h, dT)


def tas2eas(v_tas, h, dT=0):
    """Convert true airspeed to equivalent airspeed."""
    return _default_aero.tas2eas(v_tas, h, dT)


def cas2tas(v_cas, h, dT=0):
    """Convert calibrated airspeed to true airspeed."""
    return _default_aero.cas2tas(v_cas, h, dT)


def tas2cas(v_tas, h, dT=0):
    """Convert true airspeed to calibrated airspeed."""
    return _default_aero.tas2cas(v_tas, h, dT)


def mach2cas(mach, h, dT=0):
    """Convert Mach number to calibrated airspeed."""
    return _default_aero.mach2cas(mach, h, dT)


def cas2mach(v_cas, h, dT=0):
    """Convert calibrated airspeed to Mach number."""
    return _default_aero.cas2mach(v_cas, h, dT)


def crossover_alt(v_cas, mach, dT=0):
    """Compute crossover altitude for constant CAS and Mach."""
    return _default_aero.crossover_alt(v_cas, mach, dT)
