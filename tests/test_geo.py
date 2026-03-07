"""Comprehensive tests for the geo module.

This module tests all geographic calculation functions in openap.geo,
including distance, bearing, latlon, and solar zenith angle calculations.
"""

from datetime import datetime

import numpy as np
import pytest

from openap import geo
from openap.backends import CasadiBackend, JaxBackend, NumpyBackend
from openap.geo import Geo


# Tolerance for floating point comparisons
RTOL = 1e-4  # 0.01% relative tolerance

# Expected values
EXPECTED = {
    # Distance: London (51.5, -0.1) to Paris (48.85, 2.35)
    "distance_london_paris": 342400.75,  # meters (~342.4 km)
    # Bearing: London to Paris
    "bearing_london_paris": 148.4226,  # degrees
}


class TestGeoDistance:
    """Tests for distance calculations."""

    def test_distance_london_paris(self):
        """Test Haversine distance calculation."""
        geo_obj = Geo()

        # London to Paris
        lat1, lon1 = 51.5, -0.1  # London
        lat2, lon2 = 48.85, 2.35  # Paris

        dist = geo_obj.distance(lat1, lon1, lat2, lon2)
        assert dist == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_distance_same_point(self):
        """Test distance between same point is zero."""
        geo_obj = Geo()
        dist = geo_obj.distance(51.5, -0.1, 51.5, -0.1)
        assert dist == pytest.approx(0.0, abs=1.0)

    def test_distance_with_altitude(self):
        """Test distance calculation at altitude."""
        geo_obj = Geo()

        lat1, lon1 = 51.5, -0.1
        lat2, lon2 = 48.85, 2.35

        dist_ground = geo_obj.distance(lat1, lon1, lat2, lon2, h=0)
        dist_altitude = geo_obj.distance(lat1, lon1, lat2, lon2, h=10000)

        # Distance at altitude should be slightly larger (larger radius)
        assert dist_altitude > dist_ground

    def test_distance_array(self):
        """Test distance calculation with array inputs."""
        geo_obj = Geo()

        lat1 = np.array([51.5, 40.7])
        lon1 = np.array([-0.1, -74.0])
        lat2 = np.array([48.85, 34.05])
        lon2 = np.array([2.35, -118.25])

        dist = geo_obj.distance(lat1, lon1, lat2, lon2)

        assert isinstance(dist, np.ndarray)
        assert dist.shape == (2,)
        assert dist[0] == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)


class TestGeoBearing:
    """Tests for bearing calculations."""

    def test_bearing_london_paris(self):
        """Test bearing calculation."""
        geo_obj = Geo()

        # London to Paris (should be roughly south-southeast)
        lat1, lon1 = 51.5, -0.1  # London
        lat2, lon2 = 48.85, 2.35  # Paris

        brg = geo_obj.bearing(lat1, lon1, lat2, lon2)
        assert brg == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_bearing_north(self):
        """Test bearing due north."""
        geo_obj = Geo()
        brg = geo_obj.bearing(0, 0, 10, 0)
        assert brg == pytest.approx(0.0, abs=0.1)

    def test_bearing_east(self):
        """Test bearing due east."""
        geo_obj = Geo()
        brg = geo_obj.bearing(0, 0, 0, 10)
        assert brg == pytest.approx(90.0, abs=0.1)

    def test_bearing_south(self):
        """Test bearing due south."""
        geo_obj = Geo()
        brg = geo_obj.bearing(10, 0, 0, 0)
        assert brg == pytest.approx(180.0, abs=0.1)

    def test_bearing_west(self):
        """Test bearing due west."""
        geo_obj = Geo()
        brg = geo_obj.bearing(0, 10, 0, 0)
        assert brg == pytest.approx(270.0, abs=0.1)


class TestGeoLatlon:
    """Tests for latlon calculations."""

    def test_latlon_forward(self):
        """Test lat/lon calculation given distance and bearing."""
        geo_obj = Geo()

        # Start at origin, go 111km north (roughly 1 degree latitude)
        lat1, lon1 = 0.0, 0.0
        d = 111000  # meters (roughly 1 degree at equator)
        brg = 0  # north

        lat2, lon2 = geo_obj.latlon(lat1, lon1, d, brg)

        # Should be approximately 1 degree north
        assert lat2 == pytest.approx(1.0, rel=0.01)
        assert lon2 == pytest.approx(0.0, abs=0.01)

    def test_latlon_east(self):
        """Test lat/lon calculation going east."""
        geo_obj = Geo()

        lat1, lon1 = 0.0, 0.0
        d = 111000  # meters
        brg = 90  # east

        lat2, lon2 = geo_obj.latlon(lat1, lon1, d, brg)

        # Should be approximately 1 degree east
        assert lat2 == pytest.approx(0.0, abs=0.01)
        assert lon2 == pytest.approx(1.0, rel=0.01)

    def test_latlon_roundtrip(self):
        """Test that distance and latlon are consistent."""
        geo_obj = Geo()

        lat1, lon1 = 51.5, -0.1
        d = 100000  # 100 km
        brg = 45  # northeast

        lat2, lon2 = geo_obj.latlon(lat1, lon1, d, brg)

        # Calculate distance back - should match original
        d_calc = geo_obj.distance(lat1, lon1, lat2, lon2)
        assert d_calc == pytest.approx(d, rel=0.001)


class TestGeoSolarPosition:
    """Tests for solar zenith angle calculations."""

    def test_solar_zenith_angle_noon_equator_equinox(self):
        """Test solar zenith angle at solar noon on equator during equinox."""
        geo_obj = Geo()

        # March 20, 2024 at ~12:00 UTC on the equator at prime meridian
        # Sun should be nearly overhead (zenith ~0)
        timestamp = datetime(2024, 3, 20, 12, 0, 0)
        lat, lon = 0.0, 0.0

        zenith = geo_obj.solar_zenith_angle(lat, lon, timestamp)

        # At equinox, sun is directly overhead at equator at noon
        # Allow some tolerance for equation of time
        assert zenith == pytest.approx(0.0, abs=5.0)

    def test_solar_zenith_angle_midnight(self):
        """Test solar zenith angle at midnight (should be > 90)."""
        geo_obj = Geo()

        # Midnight UTC at prime meridian
        timestamp = datetime(2024, 6, 21, 0, 0, 0)
        lat, lon = 45.0, 0.0

        zenith = geo_obj.solar_zenith_angle(lat, lon, timestamp)

        # At midnight, sun should be below horizon (zenith > 90)
        assert zenith > 90

    def test_solar_zenith_angle_summer_solstice(self):
        """Test solar zenith angle on summer solstice."""
        geo_obj = Geo()

        # June 21, 2024 at solar noon at Tropic of Cancer
        timestamp = datetime(2024, 6, 21, 12, 0, 0)
        lat, lon = 23.44, 0.0  # Tropic of Cancer

        zenith = geo_obj.solar_zenith_angle(lat, lon, timestamp)

        # Sun should be nearly overhead
        assert zenith < 10

    def test_solar_zenith_angle_array_input(self):
        """Test solar zenith angle with array inputs."""
        geo_obj = Geo()

        # Multiple locations at same time - use equinox for monotonic increase
        timestamps = [datetime(2024, 3, 20, 12, 0, 0)] * 3
        lats = np.array([0.0, 30.0, 60.0])
        lons = np.array([0.0, 0.0, 0.0])

        zenith = geo_obj.solar_zenith_angle(lats, lons, timestamps)

        # Should return array
        assert len(zenith) == 3
        # At equinox, zenith angle should increase with latitude from equator
        assert zenith[0] < zenith[1] < zenith[2]


class TestGeoModuleFunctions:
    """Tests for module-level wrapper functions."""

    def test_module_distance(self):
        """Test module-level distance function."""
        dist = geo.distance(51.5, -0.1, 48.85, 2.35)
        assert dist == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_module_bearing(self):
        """Test module-level bearing function."""
        brg = geo.bearing(51.5, -0.1, 48.85, 2.35)
        assert brg == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_module_latlon(self):
        """Test module-level latlon function."""
        lat2, lon2 = geo.latlon(0, 0, 111000, 0)  # Go 111km north
        assert lat2 == pytest.approx(1.0, rel=0.01)

    def test_module_solar_zenith_angle(self):
        """Test module-level solar_zenith_angle function."""
        timestamp = datetime(2024, 3, 20, 12, 0, 0)
        zenith = geo.solar_zenith_angle(0.0, 0.0, timestamp)
        assert zenith == pytest.approx(0.0, abs=5.0)


class TestGeoCasadiBackend:
    """Tests for geo functions with CasADi backend."""

    @pytest.fixture
    def casadi(self):
        """Import casadi if available."""
        return pytest.importorskip("casadi")

    def test_distance_symbolic(self, casadi):
        """Test distance calculation with symbolic inputs."""
        geo_obj = Geo(backend=CasadiBackend())

        lat1 = casadi.SX.sym("lat1")
        lon1 = casadi.SX.sym("lon1")
        lat2 = casadi.SX.sym("lat2")
        lon2 = casadi.SX.sym("lon2")

        dist = geo_obj.distance(lat1, lon1, lat2, lon2)
        assert isinstance(dist, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [lat1, lon1, lat2, lon2], [dist])
        result = float(f(51.5, -0.1, 48.85, 2.35))
        assert result == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_bearing_symbolic(self, casadi):
        """Test bearing calculation with symbolic inputs."""
        geo_obj = Geo(backend=CasadiBackend())

        lat1 = casadi.SX.sym("lat1")
        lon1 = casadi.SX.sym("lon1")
        lat2 = casadi.SX.sym("lat2")
        lon2 = casadi.SX.sym("lon2")

        brg = geo_obj.bearing(lat1, lon1, lat2, lon2)
        assert isinstance(brg, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [lat1, lon1, lat2, lon2], [brg])
        result = float(f(51.5, -0.1, 48.85, 2.35))
        assert result == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_latlon_symbolic(self, casadi):
        """Test latlon calculation with symbolic inputs."""
        geo_obj = Geo(backend=CasadiBackend())

        lat1 = casadi.SX.sym("lat1")
        lon1 = casadi.SX.sym("lon1")
        d = casadi.SX.sym("d")
        brg = casadi.SX.sym("brg")

        lat2, lon2 = geo_obj.latlon(lat1, lon1, d, brg)
        assert isinstance(lat2, casadi.SX)
        assert isinstance(lon2, casadi.SX)


class TestGeoJaxBackend:
    """Tests for geo functions with JAX backend."""

    @pytest.fixture
    def jax(self):
        """Import jax if available."""
        return pytest.importorskip("jax")

    @pytest.fixture
    def jnp(self, jax):
        """Import jax.numpy."""
        return jax.numpy

    def test_distance_jax(self, jnp):
        """Test distance calculation with JAX."""
        geo_obj = Geo(backend=JaxBackend())

        dist = geo_obj.distance(
            jnp.array(51.5),
            jnp.array(-0.1),
            jnp.array(48.85),
            jnp.array(2.35),
        )
        assert float(dist) == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_bearing_jax(self, jnp):
        """Test bearing calculation with JAX."""
        geo_obj = Geo(backend=JaxBackend())

        brg = geo_obj.bearing(
            jnp.array(51.5),
            jnp.array(-0.1),
            jnp.array(48.85),
            jnp.array(2.35),
        )
        assert float(brg) == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_jit_geo_functions(self, jax, jnp):
        """Test JIT compilation of geo functions."""
        geo_obj = Geo(backend=JaxBackend())

        @jax.jit
        def compute_distance(lat1, lon1, lat2, lon2):
            return geo_obj.distance(lat1, lon1, lat2, lon2)

        dist = compute_distance(
            jnp.array(51.5),
            jnp.array(-0.1),
            jnp.array(48.85),
            jnp.array(2.35),
        )
        assert float(dist) == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)


class TestGeoBackwardCompatibility:
    """Tests for backward compatibility with aero module."""

    def test_aero_distance_still_works(self):
        """Test that aero.distance still works (backward compat)."""
        from openap import aero

        dist = aero.distance(51.5, -0.1, 48.85, 2.35)
        assert dist == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_aero_bearing_still_works(self):
        """Test that aero.bearing still works (backward compat)."""
        from openap import aero

        brg = aero.bearing(51.5, -0.1, 48.85, 2.35)
        assert brg == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_aero_latlon_still_works(self):
        """Test that aero.latlon still works (backward compat)."""
        from openap import aero

        lat2, lon2 = aero.latlon(0, 0, 111000, 0)
        assert lat2 == pytest.approx(1.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
