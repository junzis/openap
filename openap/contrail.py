"""Contrail formation and radiative forcing calculations.

This module provides functions for:
- Saturation pressure calculations (Murphy and Koop 2005)
- Relative humidity conversions
- Critical temperature calculations (Schmidt-Appleman criterion)
- Radiative forcing from contrails (shortwave and longwave)
- Contrail optical property evolution over time

Usage:
    from openap import contrail

    # Check contrail formation conditions
    rhi = contrail.relative_humidity(q, p, T, to="ice")
    t_crit = contrail.critical_temperature_water(p)

    # Calculate radiative forcing
    rf_sw = contrail.rf_shortwave(zenith, tau, tau_c)
    rf_lw = contrail.rf_longwave(olr, temperature)
"""

import numpy as np
from scipy import optimize

# =============================================================================
# Physical constants
# =============================================================================

gas_constant_water_vapor = 461.51  # J/(kg·K)
gas_constant_dry_air = 287.05  # J/(kg·K)
temperature_steam = 372.15  # K
pressure_steam = 101325  # Pa
temperature_ice_point = 273.16  # K
pressure_ice_point = 611.73  # Pa

# Contrail formation parameters
ei_water = 1.2232  # Water vapor emission index (kg/kg fuel)
spec_air_heat_capacity = 1004  # J/(kg·K)
ratio_mass_water_vapor_air = 0.622  # Molecular mass ratio
spec_combustion_heat = 43e6  # J/kg, specific combustion heat of jet fuel

# Default propulsion efficiency
DEFAULT_PROPULSION_EFFICIENCY = 0.4


# =============================================================================
# Saturation pressure functions (Murphy and Koop 2005)
# =============================================================================


def saturation_pressure_over_water(temperature):
    """Calculate saturation vapor pressure over liquid water.

    Uses the Murphy and Koop (2005) formulation, valid for temperatures
    from 123 K to 332 K.

    Args:
        temperature: Air temperature (K). Can be scalar or array.

    Returns:
        Saturation vapor pressure over water (Pa).

    Reference:
        Murphy, D. M. and Koop, T. (2005). Review of the vapour pressures
        of ice and supercooled water for atmospheric applications.
        Q. J. R. Meteorol. Soc., 131, 1539-1565.
    """
    return np.exp(
        54.842763
        - 6763.22 / temperature
        - 4.210 * np.log(temperature)
        + 0.000367 * temperature
        + np.tanh(0.0415 * (temperature - 218.8))
        * (
            53.878
            - 1331.22 / temperature
            - 9.44523 * np.log(temperature)
            + 0.014025 * temperature
        )
    )


def saturation_pressure_over_ice(temperature):
    """Calculate saturation vapor pressure over ice.

    Uses the Murphy and Koop (2005) formulation, valid for temperatures
    from 110 K to 273 K.

    Args:
        temperature: Air temperature (K). Can be scalar or array.

    Returns:
        Saturation vapor pressure over ice (Pa).

    Reference:
        Murphy, D. M. and Koop, T. (2005). Review of the vapour pressures
        of ice and supercooled water for atmospheric applications.
        Q. J. R. Meteorol. Soc., 131, 1539-1565.
    """
    return np.exp(
        9.550426
        - 5723.265 / temperature
        + 3.53068 * np.log(temperature)
        - 0.00728332 * temperature
    )


# =============================================================================
# Relative humidity functions
# =============================================================================


def relative_humidity(specific_humidity, pressure, temperature, to="ice"):
    """Calculate relative humidity from specific humidity.

    Args:
        specific_humidity: Specific humidity (kg/kg).
        pressure: Air pressure (Pa).
        temperature: Air temperature (K).
        to: Reference phase, either "ice" or "water". Defaults to "ice".

    Returns:
        Relative humidity (dimensionless, where 1.0 = 100%).

    Raises:
        AssertionError: If `to` is not "ice" or "water".
    """
    assert to in ("ice", "water")

    if to == "ice":
        saturation_pressure = saturation_pressure_over_ice(temperature)
    else:
        saturation_pressure = saturation_pressure_over_water(temperature)

    return (
        specific_humidity
        * pressure
        * (gas_constant_water_vapor / gas_constant_dry_air)
        / saturation_pressure
    )


def rhw2rhi(relative_humidity_water, temperature):
    """Convert relative humidity with respect to water to ice.

    Args:
        relative_humidity_water: Relative humidity w.r.t. water (dimensionless).
        temperature: Air temperature (K).

    Returns:
        Relative humidity with respect to ice (dimensionless).
    """
    return (
        relative_humidity_water
        * saturation_pressure_over_water(temperature)
        / saturation_pressure_over_ice(temperature)
    )


# =============================================================================
# Critical temperature functions (Schmidt-Appleman criterion)
# =============================================================================


def _isobaric_mixing_slope(pressure, propulsion_efficiency):
    """Calculate the isobaric mixing slope (G) for the Schmidt-Appleman criterion.

    Args:
        pressure: Air pressure (Pa).
        propulsion_efficiency: Overall propulsion efficiency (0-1).

    Returns:
        Isobaric mixing slope (Pa/K).
    """
    return (
        ei_water
        * spec_air_heat_capacity
        * pressure
        / (
            ratio_mass_water_vapor_air
            * spec_combustion_heat
            * (1 - propulsion_efficiency)
        )
    )


def critical_temperature_water(pressure, propulsion_efficiency=DEFAULT_PROPULSION_EFFICIENCY):
    """Calculate critical temperature for contrail formation (water saturation).

    This is the threshold temperature below which contrails can form,
    based on the Schmidt-Appleman criterion with water saturation.

    Args:
        pressure: Air pressure (Pa). Can be scalar or array.
        propulsion_efficiency: Overall propulsion efficiency (0-1).
            Defaults to 0.4. Higher efficiency means higher critical
            temperature (contrails form more easily).

    Returns:
        Critical temperature (K). Contrails can form when the ambient
        temperature is below this value.

    Reference:
        Schumann, U. (1996). On conditions for contrail formation from
        aircraft exhausts. Meteorol. Z., 5, 4-23.
    """
    g = _isobaric_mixing_slope(pressure, propulsion_efficiency)

    crit_temp_water = (
        -46.46
        + 9.43 * np.log(g - 0.053)
        + 0.72 * (np.log(g - 0.053)) ** 2
        + 273.15
    )

    return crit_temp_water


def critical_temperature_water_and_ice(pressure, propulsion_efficiency=DEFAULT_PROPULSION_EFFICIENCY):
    """Calculate critical temperatures for contrail formation (water and ice).

    Returns both the water saturation critical temperature and the
    ice saturation critical temperature for contrail persistence.

    Args:
        pressure: Air pressure (Pa). Scalar only (due to root finding).
        propulsion_efficiency: Overall propulsion efficiency (0-1).
            Defaults to 0.4.

    Returns:
        Tuple of (crit_temp_water, crit_temp_ice) in Kelvin.
        - crit_temp_water: Temperature below which contrails can form
        - crit_temp_ice: Temperature below which contrails can persist
          (if ice supersaturation exists)

    Reference:
        Schumann, U. (1996). On conditions for contrail formation from
        aircraft exhausts. Meteorol. Z., 5, 4-23.
    """
    def func(temp_critical, crit_temp_water, g):
        return (
            saturation_pressure_over_water(crit_temp_water)
            - saturation_pressure_over_ice(temp_critical)
            - (crit_temp_water - temp_critical) * g
        )

    g = _isobaric_mixing_slope(pressure, propulsion_efficiency)

    crit_temp_water = (
        -46.46
        + 9.43 * np.log(g - 0.053)
        + 0.72 * (np.log(g - 0.053)) ** 2
        + 273.15
    )

    sol = optimize.root_scalar(
        func,
        args=(crit_temp_water, g),
        bracket=[100, crit_temp_water],
        method="brentq",
    )
    crit_temp_ice = sol.root

    return crit_temp_water, crit_temp_ice


# =============================================================================
# Radiative forcing functions
# =============================================================================


def rf_shortwave(zenith, tau, tau_c):
    """Calculate shortwave radiative forcing from contrails.

    Computes the shortwave (solar) radiative forcing based on contrail
    optical properties and solar geometry. Shortwave forcing is typically
    negative (cooling effect) during daytime.

    Args:
        zenith: Solar zenith angle (degrees). 0 = sun overhead,
            90 = horizon, >90 = nighttime.
        tau: Contrail optical depth (dimensionless).
        tau_c: Background cirrus optical depth (dimensionless).

    Returns:
        Shortwave radiative forcing (W/m²). Negative values indicate
        cooling (reflection of incoming solar radiation).
        Returns 0 for nighttime (zenith > 90).
    """
    # Optical parameters
    tA = 0.879  # Atmospheric transmittance
    gamma_val = 0.242  # Contrail reflectance parameter
    gamma_l = 0.323  # Contrail reflectance parameter (large angles)
    A_mu = 0.361  # Angular parameter
    B_mu = 1.676  # Angular parameter
    C_mu = 0.709  # Angular parameter

    # Physical parameters
    Fr = 0.512  # Fraction parameter
    delta_SR = 0.15  # Scattering parameter
    delta_SC = 0.157  # Cirrus scattering parameter
    delta_app_SC = 0.23  # Apparent cirrus scattering
    S0 = 1361  # Solar constant (W/m²)
    reff = 16  # Effective particle radius (μm)
    Aeff = 0.2  # Effective albedo

    # Shortwave fraction
    Fsw = 1 - (Fr * (1 - np.exp(-delta_SR * reff)))

    # Solar direct radiation and cosine of zenith
    SDR = np.cos(np.deg2rad(zenith)) * S0
    mu = np.cos(np.deg2rad(zenith))

    # Effective optical depths
    tau_eff = (tau * Fsw) / mu
    tau_c_eff = (tau_c * Fsw) / mu

    # Angular function
    F_mu = (((1 - mu) ** B_mu) / ((1 / 2) ** B_mu)) - 1

    # Contrail reflectance
    R_c = 1 - np.exp(-gamma_val * tau_eff)
    alpha_c = R_c * (C_mu + (A_mu * np.exp(-gamma_l * tau_eff) * F_mu))

    # Enhancement factor from cirrus
    E_sw = np.exp((delta_SC * tau_c) - (delta_app_SC * tau_c_eff))

    # Shortwave forcing (negative = cooling)
    rf = -SDR * ((tA - Aeff) ** 2) * alpha_c * E_sw

    # Return 0 for nighttime
    return np.where(zenith > 90, 0, rf)


def rf_longwave(olr, temperature):
    """Calculate longwave radiative forcing from contrails.

    Computes the longwave (thermal infrared) radiative forcing based on
    outgoing longwave radiation and contrail temperature. Longwave forcing
    is typically positive (warming effect).

    Args:
        olr: Outgoing longwave radiation at top of atmosphere (W/m²).
            Can be obtained from satellite data or reanalysis.
        temperature: Contrail/ambient temperature (K).

    Returns:
        Longwave radiative forcing (W/m²). Positive values indicate
        warming (trapping of outgoing thermal radiation).
    """
    kt = 1.935  # Temperature sensitivity parameter
    T0 = 152  # Reference temperature (K)

    return np.maximum(olr - kt * (temperature - T0), 0)


def rf_net(zenith, tau, tau_c, olr, temperature):
    """Calculate net radiative forcing from contrails.

    Combines shortwave (cooling) and longwave (warming) forcing to
    compute the net climate impact.

    Args:
        zenith: Solar zenith angle (degrees).
        tau: Contrail optical depth (dimensionless).
        tau_c: Background cirrus optical depth (dimensionless).
        olr: Outgoing longwave radiation (W/m²).
        temperature: Contrail/ambient temperature (K).

    Returns:
        Net radiative forcing (W/m²). Positive = warming, negative = cooling.
    """
    sw = rf_shortwave(zenith, tau, tau_c)
    lw = rf_longwave(olr, temperature)
    return sw + lw


def load_olr(filepath, lat, lon, time):
    """Load OLR data from a netCDF file.

    Interpolates outgoing longwave radiation data to specified locations
    and times. Requires xarray (optional dependency).

    Args:
        filepath: Path to netCDF file containing OLR data.
            Expected variables: 'olr' or 'OLR', with dimensions
            (time, lat/latitude, lon/longitude).
        lat: Latitude(s) to interpolate (degrees).
        lon: Longitude(s) to interpolate (degrees).
        time: Timestamp(s) for temporal interpolation.
            Can be datetime, pandas Timestamp, or array of timestamps.

    Returns:
        OLR values (W/m²) at the specified locations and times.

    Raises:
        ImportError: If xarray is not installed.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for load_olr(). "
            "Install with: pip install xarray"
        )

    ds = xr.open_dataset(filepath)

    # Find OLR variable (common names)
    olr_var = None
    for name in ["olr", "OLR", "olr_avg", "toa_lw_all_daily"]:
        if name in ds:
            olr_var = name
            break

    if olr_var is None:
        raise ValueError(
            f"Could not find OLR variable in {filepath}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    # Find coordinate names
    lat_coord = None
    for name in ["lat", "latitude"]:
        if name in ds.coords:
            lat_coord = name
            break

    lon_coord = None
    for name in ["lon", "longitude"]:
        if name in ds.coords:
            lon_coord = name
            break

    # Interpolate
    result = ds[olr_var].interp(
        {lat_coord: lat, lon_coord: lon, "time": time},
        method="linear",
    )

    return result.values


# =============================================================================
# Contrail optical evolution
# =============================================================================


def contrail_optical_properties(age_hours):
    """Get contrail optical properties based on age.

    Models contrail evolution from young thin trails to aged artificial
    cirrus. Properties are based on typical contrail lifecycle observations.

    Optical property evolution (cumulative values):

        Age (hours) | tau   | width (m) | tau_c
        ------------|-------|-----------|-------
        0-1         | 0.40  | 500       | 0.360
        1-2         | 0.60  | 1500      | 0.540
        2-4         | 0.68  | 3500      | 0.612
        4-6         | 0.70  | 6500      | 0.630
        6+          | 0.71  | 10500     | 0.639

    Args:
        age_hours: Contrail age in hours. Can be scalar or array.

    Returns:
        Tuple of (tau, width_m, tau_c):
        - tau: Contrail optical depth (dimensionless)
        - width_m: Contrail width (meters)
        - tau_c: Cirrus optical depth (dimensionless)
    """
    age = np.asarray(age_hours)

    # Cumulative values at hour boundaries
    tau_values = np.array([0.4, 0.6, 0.68, 0.70, 0.71])
    width_values = np.array([500, 1500, 3500, 6500, 10500])
    tau_c_values = np.array([0.36, 0.54, 0.612, 0.63, 0.639])

    # Hour boundaries
    hour_bounds = np.array([1, 2, 4, 6])

    # Use searchsorted to find the appropriate bin
    indices = np.searchsorted(hour_bounds, age, side="right")
    indices = np.clip(indices, 0, len(tau_values) - 1)

    tau = tau_values[indices]
    width = width_values[indices]
    tau_c = tau_c_values[indices]

    # Return scalars if input was scalar
    if np.ndim(age_hours) == 0:
        return float(tau), float(width), float(tau_c)

    return tau, width, tau_c


# Backward compatibility - keep the module-level constant
propulsion_efficiency = DEFAULT_PROPULSION_EFFICIENCY
