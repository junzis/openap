"""Geographic and navigation calculations with backend support.

This module provides geographic functions (distance, bearing, position)
and solar position calculations that work with any backend (NumPy, CasADi, JAX).

For backward compatibility, module-level functions use NumPy.
For other backends, use the Geo class directly.

Usage:
    # NumPy (default) - module functions
    from openap import geo
    dist = geo.distance(51.5, -0.1, 48.85, 2.35)

    # With specific backend
    from openap.geo import Geo
    from openap.backends import JaxBackend
    geo_jax = Geo(backend=JaxBackend())
    dist = geo_jax.distance(51.5, -0.1, 48.85, 2.35)
"""

from datetime import datetime
from typing import Any, Optional, Tuple, Union

from openap.backends import BackendType, NumpyBackend, default_backend

# =============================================================================
# Constants
# =============================================================================

r_earth = 6371000.0  # m, average earth radius


# =============================================================================
# Geo class with backend support
# =============================================================================


class Geo:
    """Geographic calculations with configurable backend.

    This class provides geographic and navigation functions
    using the specified math backend.

    Args:
        backend: Math backend to use. Defaults to NumpyBackend.
    """

    r_earth = r_earth

    def __init__(self, backend: Optional[BackendType] = None):
        """Initialize Geo with a backend.

        Args:
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        self.backend = backend or default_backend()

    def distance(self, lat1: Any, lon1: Any, lat2: Any, lon2: Any, h: Any = 0) -> Any:
        """Compute distance between coordinates using Haversine formula.

        Args:
            lat1: Starting latitude (degrees).
            lon1: Starting longitude (degrees).
            lat2: Ending latitude (degrees).
            lon2: Ending longitude (degrees).
            h: Altitude (meters). Defaults to 0.

        Returns:
            Distance (meters).
        """
        b = self.backend
        deg2rad = b.pi / 180.0

        lat1_r = lat1 * deg2rad
        lon1_r = lon1 * deg2rad
        lat2_r = lat2 * deg2rad
        lon2_r = lon2 * deg2rad

        dlon = lon2_r - lon1_r
        dlat = lat2_r - lat1_r

        a = b.sin(dlat / 2) ** 2 + b.cos(lat1_r) * b.cos(lat2_r) * b.sin(dlon / 2) ** 2
        c = 2 * b.arcsin(b.sqrt(a))
        dist = c * (r_earth + h)
        return dist

    def bearing(self, lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
        """Compute initial bearing between coordinates.

        Args:
            lat1: Starting latitude (degrees).
            lon1: Starting longitude (degrees).
            lat2: Ending latitude (degrees).
            lon2: Ending longitude (degrees).

        Returns:
            Bearing (degrees). Between 0 and 360.
        """
        b = self.backend
        deg2rad = b.pi / 180.0
        rad2deg = 180.0 / b.pi

        lat1_r = lat1 * deg2rad
        lon1_r = lon1 * deg2rad
        lat2_r = lat2 * deg2rad
        lon2_r = lon2 * deg2rad

        x = b.sin(lon2_r - lon1_r) * b.cos(lat2_r)
        y = b.cos(lat1_r) * b.sin(lat2_r) - b.sin(lat1_r) * b.cos(lat2_r) * b.cos(
            lon2_r - lon1_r
        )
        initial_bearing = b.arctan2(x, y) * rad2deg
        return b.fmod(initial_bearing + 360, 360)

    def latlon(
        self, lat1: Any, lon1: Any, d: Any, brg: Any, h: Any = 0
    ) -> Tuple[Any, Any]:
        """Compute destination point given start, distance and bearing.

        Args:
            lat1: Starting latitude (degrees).
            lon1: Starting longitude (degrees).
            d: Distance from point 1 (meters).
            brg: Bearing at point 1 (degrees).
            h: Altitude (meters). Defaults to 0.

        Returns:
            Tuple of (latitude, longitude) in degrees.
        """
        b = self.backend
        deg2rad = b.pi / 180.0
        rad2deg = 180.0 / b.pi

        lat1_r = lat1 * deg2rad
        lon1_r = lon1 * deg2rad
        brg_r = brg * deg2rad

        lat2 = b.arcsin(
            b.sin(lat1_r) * b.cos(d / (r_earth + h))
            + b.cos(lat1_r) * b.sin(d / (r_earth + h)) * b.cos(brg_r)
        )
        lon2 = lon1_r + b.arctan2(
            b.sin(brg_r) * b.sin(d / (r_earth + h)) * b.cos(lat1_r),
            b.cos(d / (r_earth + h)) - b.sin(lat1_r) * b.sin(lat2),
        )

        return lat2 * rad2deg, lon2 * rad2deg

    def solar_zenith_angle(
        self,
        lat: Any,
        lon: Any,
        timestamp: Union[datetime, Any],
    ) -> Any:
        """Calculate solar zenith angle.

        Uses the standard astronomical formula for solar position.

        Args:
            lat: Latitude in degrees (-90 to 90).
            lon: Longitude in degrees (-180 to 180).
            timestamp: UTC timestamp. Can be datetime, pandas Timestamp,
                or array of timestamps.

        Returns:
            Solar zenith angle in degrees (0 = sun overhead, 90 = horizon,
            >90 = night).
        """
        import numpy as np

        # Convert timestamp to day of year and fractional hour
        if hasattr(timestamp, "__iter__") and not isinstance(timestamp, str):
            # Array of timestamps
            timestamps = np.asarray(timestamp)
            if hasattr(timestamps[0], "timetuple"):
                day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])
                hour_utc = np.array(
                    [t.hour + t.minute / 60 + t.second / 3600 for t in timestamps]
                )
            else:
                # Assume unix timestamp
                from datetime import timezone

                dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
                day_of_year = np.array([dt.timetuple().tm_yday for dt in dts])
                hour_utc = np.array(
                    [dt.hour + dt.minute / 60 + dt.second / 3600 for dt in dts]
                )
        else:
            # Single timestamp
            if hasattr(timestamp, "timetuple"):
                day_of_year = timestamp.timetuple().tm_yday
                hour_utc = (
                    timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
                )
            else:
                # Assume unix timestamp
                from datetime import timezone

                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                day_of_year = dt.timetuple().tm_yday
                hour_utc = dt.hour + dt.minute / 60 + dt.second / 3600

        # Solar declination (Spencer, 1971)
        gamma = 2 * np.pi * (day_of_year - 1) / 365

        declination = (
            0.006918
            - 0.399912 * np.cos(gamma)
            + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma)
            + 0.00148 * np.sin(3 * gamma)
        )

        # Equation of time (minutes)
        eot = 229.18 * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma)
        )

        # Convert inputs to radians
        lat_rad = np.deg2rad(lat)
        lon_np = np.asarray(lon)

        # Solar time
        time_offset = eot + 4 * lon_np  # minutes
        solar_time = hour_utc * 60 + time_offset  # minutes from midnight

        # Hour angle (degrees, then radians)
        hour_angle_deg = (solar_time / 4) - 180  # degrees
        hour_angle = np.deg2rad(hour_angle_deg)

        # Zenith angle calculation
        cos_zenith = np.sin(lat_rad) * np.sin(declination) + np.cos(
            lat_rad
        ) * np.cos(declination) * np.cos(hour_angle)

        # Clamp to [-1, 1] to avoid numerical issues with arccos
        cos_zenith = np.clip(cos_zenith, -1, 1)

        zenith = np.rad2deg(np.arccos(cos_zenith))

        return zenith


# =============================================================================
# Module-level functions for backward compatibility (using NumPy)
# =============================================================================

_default_geo = Geo(backend=NumpyBackend())


def distance(lat1, lon1, lat2, lon2, h=0):
    """Compute distance between coordinates using Haversine formula.

    Args:
        lat1: Starting latitude (degrees).
        lon1: Starting longitude (degrees).
        lat2: Ending latitude (degrees).
        lon2: Ending longitude (degrees).
        h: Altitude (meters). Defaults to 0.

    Returns:
        Distance (meters).
    """
    return _default_geo.distance(lat1, lon1, lat2, lon2, h)


def bearing(lat1, lon1, lat2, lon2):
    """Compute initial bearing between coordinates.

    Args:
        lat1: Starting latitude (degrees).
        lon1: Starting longitude (degrees).
        lat2: Ending latitude (degrees).
        lon2: Ending longitude (degrees).

    Returns:
        Bearing (degrees). Between 0 and 360.
    """
    return _default_geo.bearing(lat1, lon1, lat2, lon2)


def latlon(lat1, lon1, d, brg, h=0):
    """Compute destination point given start, distance and bearing.

    Args:
        lat1: Starting latitude (degrees).
        lon1: Starting longitude (degrees).
        d: Distance from point 1 (meters).
        brg: Bearing at point 1 (degrees).
        h: Altitude (meters). Defaults to 0.

    Returns:
        Tuple of (latitude, longitude) in degrees.
    """
    return _default_geo.latlon(lat1, lon1, d, brg, h)


def solar_zenith_angle(lat, lon, timestamp):
    """Calculate solar zenith angle.

    Uses the standard astronomical formula for solar position.

    Args:
        lat: Latitude in degrees (-90 to 90).
        lon: Longitude in degrees (-180 to 180).
        timestamp: UTC timestamp. Can be datetime, pandas Timestamp,
            or array of timestamps.

    Returns:
        Solar zenith angle in degrees (0 = sun overhead, 90 = horizon,
        >90 = night).
    """
    return _default_geo.solar_zenith_angle(lat, lon, timestamp)
