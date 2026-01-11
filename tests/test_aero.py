"""Comprehensive tests for the aero module.

This module tests all aeronautical calculation functions in openap.aero,
including atmospheric properties, airspeed conversions, and navigation functions.
"""

import numpy as np
import pytest

from openap import aero
from openap.aero import Aero
from openap.backends import CasadiBackend, JaxBackend, NumpyBackend


# Tolerance for floating point comparisons
RTOL = 1e-4  # 0.01% relative tolerance

# Expected values (computed with the openap implementation)
EXPECTED = {
    # Atmospheric properties at h=10000m
    "temperature_10km": 223.15,  # K
    "density_10km": 0.412604,  # kg/m³
    "pressure_10km": 26429.70,  # Pa
    "vsound_10km": 299.4632,  # m/s
    # Atmospheric properties at h=0m (sea level)
    "temperature_0m": 288.15,  # K
    "density_0m": 1.225,  # kg/m³
    "pressure_0m": 101325.0,  # Pa
    "vsound_0m": 340.294,  # m/s
    # Stratosphere (h=15000m)
    "temperature_15km": 216.65,  # K (constant in stratosphere)
    "pressure_15km": 12041.15,  # Pa
    # Distance: London (51.5, -0.1) to Paris (48.85, 2.35)
    "distance_london_paris": 342400.75,  # meters (~342.4 km)
    # Bearing: London to Paris
    "bearing_london_paris": 148.4226,  # degrees
    # ISA altitude from pressure
    "h_isa_sealevel": 0.0,  # m (p=101325 Pa)
    "h_isa_10km": 10000.0,  # m (p≈26500 Pa)
    # Airspeed conversions at h=10000m
    "tas_from_eas": 344.6127,  # m/s (EAS=200 m/s at 10km)
    "eas_from_tas": 116.0723,  # m/s (TAS=200 m/s at 10km)
    "tas_from_cas": 314.5946,  # m/s (CAS=200 m/s at 10km)
    "cas_from_tas": 120.7405,  # m/s (TAS=200 m/s at 10km)
    "mach_from_tas": 0.6679,  # (TAS=200 m/s at 10km)
    "mach_from_cas": 1.0505,  # (CAS=200 m/s at 10km)
    # Crossover altitude (CAS=150 m/s, Mach=0.78)
    "crossover_alt": 9335.23,  # m
}


class TestAeroAtmospheric:
    """Tests for atmospheric property calculations."""

    def test_atmos_sea_level(self):
        """Test atmospheric properties at sea level."""
        aero_obj = Aero()
        p, rho, T = aero_obj.atmos(0)

        assert T == pytest.approx(EXPECTED["temperature_0m"], rel=RTOL)
        assert rho == pytest.approx(EXPECTED["density_0m"], rel=RTOL)
        assert p == pytest.approx(EXPECTED["pressure_0m"], rel=RTOL)

    def test_atmos_troposphere(self):
        """Test atmospheric properties in troposphere (10km)."""
        aero_obj = Aero()
        p, rho, T = aero_obj.atmos(10000)

        assert T == pytest.approx(EXPECTED["temperature_10km"], rel=RTOL)
        assert rho == pytest.approx(EXPECTED["density_10km"], rel=RTOL)
        assert p == pytest.approx(EXPECTED["pressure_10km"], rel=RTOL)

    def test_atmos_stratosphere(self):
        """Test atmospheric properties in stratosphere (15km)."""
        aero_obj = Aero()
        p, rho, T = aero_obj.atmos(15000)

        # Temperature is constant in stratosphere (216.65 K)
        assert T == pytest.approx(EXPECTED["temperature_15km"], rel=RTOL)
        assert p == pytest.approx(EXPECTED["pressure_15km"], rel=RTOL)

    def test_atmos_with_isa_deviation(self):
        """Test atmospheric properties with ISA temperature deviation."""
        aero_obj = Aero()

        # +10K ISA deviation
        p, rho, T = aero_obj.atmos(10000, dT=10)
        assert T == pytest.approx(223.15 + 10, rel=RTOL)

        # -10K ISA deviation
        p, rho, T = aero_obj.atmos(10000, dT=-10)
        assert T == pytest.approx(223.15 - 10, rel=RTOL)

    def test_temperature(self):
        """Test temperature calculation."""
        aero_obj = Aero()
        T = aero_obj.temperature(10000)
        assert T == pytest.approx(EXPECTED["temperature_10km"], rel=RTOL)

    def test_pressure(self):
        """Test pressure calculation."""
        aero_obj = Aero()
        p = aero_obj.pressure(10000)
        assert p == pytest.approx(EXPECTED["pressure_10km"], rel=RTOL)

    def test_density(self):
        """Test density calculation."""
        aero_obj = Aero()
        rho = aero_obj.density(10000)
        assert rho == pytest.approx(EXPECTED["density_10km"], rel=RTOL)

    def test_vsound(self):
        """Test speed of sound calculation."""
        aero_obj = Aero()
        a = aero_obj.vsound(10000)
        assert a == pytest.approx(EXPECTED["vsound_10km"], rel=RTOL)

        a_sl = aero_obj.vsound(0)
        assert a_sl == pytest.approx(EXPECTED["vsound_0m"], rel=RTOL)

    def test_h_isa_troposphere(self):
        """Test ISA altitude calculation in troposphere."""
        aero_obj = Aero()

        # Sea level pressure should give h=0
        h = aero_obj.h_isa(101325)
        assert h == pytest.approx(0.0, abs=1.0)

        # Pressure at 10km should give h≈10000
        h = aero_obj.h_isa(26429.70)
        assert h == pytest.approx(10000.0, rel=0.01)

    def test_h_isa_stratosphere(self):
        """Test ISA altitude calculation in stratosphere."""
        aero_obj = Aero()

        # Pressure in stratosphere (p < 22630 Pa)
        h = aero_obj.h_isa(12044.57)
        assert h == pytest.approx(15000.0, rel=0.01)

    def test_h_isa_with_isa_deviation(self):
        """Test ISA altitude with temperature deviation."""
        aero_obj = Aero()

        # With ISA deviation, altitude for same pressure changes
        h_std = aero_obj.h_isa(26429.70, dT=0)
        h_warm = aero_obj.h_isa(26429.70, dT=10)

        # Warmer atmosphere means same pressure at higher altitude
        assert h_warm > h_std


class TestAeroNavigation:
    """Tests for navigation functions (distance, bearing, latlon)."""

    def test_distance_london_paris(self):
        """Test Haversine distance calculation."""
        aero_obj = Aero()

        # London to Paris
        lat1, lon1 = 51.5, -0.1  # London
        lat2, lon2 = 48.85, 2.35  # Paris

        dist = aero_obj.distance(lat1, lon1, lat2, lon2)
        assert dist == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_distance_same_point(self):
        """Test distance between same point is zero."""
        aero_obj = Aero()
        dist = aero_obj.distance(51.5, -0.1, 51.5, -0.1)
        assert dist == pytest.approx(0.0, abs=1.0)

    def test_distance_with_altitude(self):
        """Test distance calculation at altitude."""
        aero_obj = Aero()

        lat1, lon1 = 51.5, -0.1
        lat2, lon2 = 48.85, 2.35

        dist_ground = aero_obj.distance(lat1, lon1, lat2, lon2, h=0)
        dist_altitude = aero_obj.distance(lat1, lon1, lat2, lon2, h=10000)

        # Distance at altitude should be slightly larger (larger radius)
        assert dist_altitude > dist_ground

    def test_bearing_london_paris(self):
        """Test bearing calculation."""
        aero_obj = Aero()

        # London to Paris (should be roughly south-southeast)
        lat1, lon1 = 51.5, -0.1  # London
        lat2, lon2 = 48.85, 2.35  # Paris

        brg = aero_obj.bearing(lat1, lon1, lat2, lon2)
        assert brg == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_bearing_north(self):
        """Test bearing due north."""
        aero_obj = Aero()
        brg = aero_obj.bearing(0, 0, 10, 0)
        assert brg == pytest.approx(0.0, abs=0.1)

    def test_bearing_east(self):
        """Test bearing due east."""
        aero_obj = Aero()
        brg = aero_obj.bearing(0, 0, 0, 10)
        assert brg == pytest.approx(90.0, abs=0.1)

    def test_bearing_south(self):
        """Test bearing due south."""
        aero_obj = Aero()
        brg = aero_obj.bearing(10, 0, 0, 0)
        assert brg == pytest.approx(180.0, abs=0.1)

    def test_bearing_west(self):
        """Test bearing due west."""
        aero_obj = Aero()
        brg = aero_obj.bearing(0, 10, 0, 0)
        assert brg == pytest.approx(270.0, abs=0.1)

    def test_latlon_forward(self):
        """Test lat/lon calculation given distance and bearing."""
        aero_obj = Aero()

        # Start at origin, go 111km north (roughly 1 degree latitude)
        lat1, lon1 = 0.0, 0.0
        d = 111000  # meters (roughly 1 degree at equator)
        brg = 0  # north

        lat2, lon2 = aero_obj.latlon(lat1, lon1, d, brg)

        # Should be approximately 1 degree north
        assert lat2 == pytest.approx(1.0, rel=0.01)
        assert lon2 == pytest.approx(0.0, abs=0.01)

    def test_latlon_east(self):
        """Test lat/lon calculation going east."""
        aero_obj = Aero()

        lat1, lon1 = 0.0, 0.0
        d = 111000  # meters
        brg = 90  # east

        lat2, lon2 = aero_obj.latlon(lat1, lon1, d, brg)

        # Should be approximately 1 degree east
        assert lat2 == pytest.approx(0.0, abs=0.01)
        assert lon2 == pytest.approx(1.0, rel=0.01)

    def test_latlon_roundtrip(self):
        """Test that distance and latlon are consistent."""
        aero_obj = Aero()

        lat1, lon1 = 51.5, -0.1
        d = 100000  # 100 km
        brg = 45  # northeast

        lat2, lon2 = aero_obj.latlon(lat1, lon1, d, brg)

        # Calculate distance back - should match original
        d_calc = aero_obj.distance(lat1, lon1, lat2, lon2)
        assert d_calc == pytest.approx(d, rel=0.001)


class TestAeroAirspeedConversions:
    """Tests for airspeed conversion functions."""

    def test_tas2mach(self):
        """Test TAS to Mach conversion."""
        aero_obj = Aero()
        mach = aero_obj.tas2mach(200, 10000)  # 200 m/s TAS at 10km
        assert mach == pytest.approx(EXPECTED["mach_from_tas"], rel=RTOL)

    def test_mach2tas(self):
        """Test Mach to TAS conversion."""
        aero_obj = Aero()
        tas = aero_obj.mach2tas(0.8, 10000)  # Mach 0.8 at 10km
        expected_tas = 0.8 * EXPECTED["vsound_10km"]
        assert tas == pytest.approx(expected_tas, rel=RTOL)

    def test_tas_mach_roundtrip(self):
        """Test TAS <-> Mach roundtrip consistency."""
        aero_obj = Aero()

        tas_orig = 250.0
        h = 10000

        mach = aero_obj.tas2mach(tas_orig, h)
        tas_back = aero_obj.mach2tas(mach, h)

        assert tas_back == pytest.approx(tas_orig, rel=RTOL)

    def test_eas2tas(self):
        """Test EAS to TAS conversion."""
        aero_obj = Aero()
        tas = aero_obj.eas2tas(200, 10000)  # 200 m/s EAS at 10km
        assert tas == pytest.approx(EXPECTED["tas_from_eas"], rel=RTOL)

    def test_tas2eas(self):
        """Test TAS to EAS conversion."""
        aero_obj = Aero()
        eas = aero_obj.tas2eas(200, 10000)  # 200 m/s TAS at 10km
        assert eas == pytest.approx(EXPECTED["eas_from_tas"], rel=RTOL)

    def test_eas_tas_roundtrip(self):
        """Test EAS <-> TAS roundtrip consistency."""
        aero_obj = Aero()

        eas_orig = 200.0
        h = 10000

        tas = aero_obj.eas2tas(eas_orig, h)
        eas_back = aero_obj.tas2eas(tas, h)

        assert eas_back == pytest.approx(eas_orig, rel=RTOL)

    def test_cas2tas(self):
        """Test CAS to TAS conversion."""
        aero_obj = Aero()
        tas = aero_obj.cas2tas(200, 10000)  # 200 m/s CAS at 10km
        assert tas == pytest.approx(EXPECTED["tas_from_cas"], rel=RTOL)

    def test_tas2cas(self):
        """Test TAS to CAS conversion."""
        aero_obj = Aero()
        cas = aero_obj.tas2cas(200, 10000)  # 200 m/s TAS at 10km
        assert cas == pytest.approx(EXPECTED["cas_from_tas"], rel=RTOL)

    def test_cas_tas_roundtrip(self):
        """Test CAS <-> TAS roundtrip consistency."""
        aero_obj = Aero()

        cas_orig = 200.0
        h = 10000

        tas = aero_obj.cas2tas(cas_orig, h)
        cas_back = aero_obj.tas2cas(tas, h)

        assert cas_back == pytest.approx(cas_orig, rel=RTOL)

    def test_cas2mach(self):
        """Test CAS to Mach conversion."""
        aero_obj = Aero()
        mach = aero_obj.cas2mach(200, 10000)  # 200 m/s CAS at 10km
        assert mach == pytest.approx(EXPECTED["mach_from_cas"], rel=RTOL)

    def test_mach2cas(self):
        """Test Mach to CAS conversion."""
        aero_obj = Aero()

        # Round trip test
        cas_orig = 200.0
        h = 10000

        mach = aero_obj.cas2mach(cas_orig, h)
        cas_back = aero_obj.mach2cas(mach, h)

        assert cas_back == pytest.approx(cas_orig, rel=RTOL)

    def test_crossover_altitude(self):
        """Test crossover altitude calculation."""
        aero_obj = Aero()

        # Typical climb: CAS=150 m/s (~290 kts), Mach=0.78
        h = aero_obj.crossover_alt(150, 0.78)
        assert h == pytest.approx(EXPECTED["crossover_alt"], rel=0.01)

    def test_crossover_altitude_consistency(self):
        """Test that CAS and Mach match at crossover altitude."""
        aero_obj = Aero()

        v_cas = 150.0
        mach = 0.78

        h_cross = aero_obj.crossover_alt(v_cas, mach)

        # At crossover, converting CAS to Mach should give target Mach
        mach_at_cross = aero_obj.cas2mach(v_cas, h_cross)
        assert mach_at_cross == pytest.approx(mach, rel=0.01)

    def test_airspeed_with_isa_deviation(self):
        """Test airspeed conversions with ISA temperature deviation."""
        aero_obj = Aero()

        # Standard conditions
        tas_std = aero_obj.cas2tas(200, 10000, dT=0)

        # Warm atmosphere (+10K)
        tas_warm = aero_obj.cas2tas(200, 10000, dT=10)

        # ISA deviation affects the result (values should differ)
        assert tas_warm != tas_std
        # The relationship depends on altitude and temperature profile
        # Just verify the function handles dT parameter correctly
        assert tas_warm > 0 and tas_std > 0


class TestAeroModuleFunctions:
    """Tests for module-level wrapper functions."""

    def test_module_atmos(self):
        """Test module-level atmos function."""
        p, rho, T = aero.atmos(10000)

        assert T == pytest.approx(EXPECTED["temperature_10km"], rel=RTOL)
        assert rho == pytest.approx(EXPECTED["density_10km"], rel=RTOL)
        assert p == pytest.approx(EXPECTED["pressure_10km"], rel=RTOL)

    def test_module_temperature(self):
        """Test module-level temperature function."""
        T = aero.temperature(10000)
        assert T == pytest.approx(EXPECTED["temperature_10km"], rel=RTOL)

    def test_module_pressure(self):
        """Test module-level pressure function."""
        p = aero.pressure(10000)
        assert p == pytest.approx(EXPECTED["pressure_10km"], rel=RTOL)

    def test_module_density(self):
        """Test module-level density function."""
        rho = aero.density(10000)
        assert rho == pytest.approx(EXPECTED["density_10km"], rel=RTOL)

    def test_module_vsound(self):
        """Test module-level vsound function."""
        a = aero.vsound(10000)
        assert a == pytest.approx(EXPECTED["vsound_10km"], rel=RTOL)

    def test_module_distance(self):
        """Test module-level distance function."""
        dist = aero.distance(51.5, -0.1, 48.85, 2.35)
        assert dist == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_module_bearing(self):
        """Test module-level bearing function."""
        brg = aero.bearing(51.5, -0.1, 48.85, 2.35)
        assert brg == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_module_h_isa(self):
        """Test module-level h_isa function."""
        h = aero.h_isa(26429.70)
        assert h == pytest.approx(10000.0, rel=0.01)

    def test_module_latlon(self):
        """Test module-level latlon function."""
        lat2, lon2 = aero.latlon(0, 0, 111000, 0)  # Go 111km north
        assert lat2 == pytest.approx(1.0, rel=0.01)

    def test_module_tas2mach(self):
        """Test module-level tas2mach function."""
        mach = aero.tas2mach(200, 10000)
        assert mach == pytest.approx(EXPECTED["mach_from_tas"], rel=RTOL)

    def test_module_mach2tas(self):
        """Test module-level mach2tas function."""
        tas = aero.mach2tas(0.8, 10000)
        expected_tas = 0.8 * EXPECTED["vsound_10km"]
        assert tas == pytest.approx(expected_tas, rel=RTOL)

    def test_module_eas2tas(self):
        """Test module-level eas2tas function."""
        tas = aero.eas2tas(200, 10000)
        assert tas == pytest.approx(EXPECTED["tas_from_eas"], rel=RTOL)

    def test_module_tas2eas(self):
        """Test module-level tas2eas function."""
        eas = aero.tas2eas(200, 10000)
        assert eas == pytest.approx(EXPECTED["eas_from_tas"], rel=RTOL)

    def test_module_cas2tas(self):
        """Test module-level cas2tas function."""
        tas = aero.cas2tas(200, 10000)
        assert tas == pytest.approx(EXPECTED["tas_from_cas"], rel=RTOL)

    def test_module_tas2cas(self):
        """Test module-level tas2cas function."""
        cas = aero.tas2cas(200, 10000)
        assert cas == pytest.approx(EXPECTED["cas_from_tas"], rel=RTOL)

    def test_module_mach2cas(self):
        """Test module-level mach2cas function."""
        # Roundtrip test
        cas_orig = 200.0
        mach = aero.cas2mach(cas_orig, 10000)
        cas_back = aero.mach2cas(mach, 10000)
        assert cas_back == pytest.approx(cas_orig, rel=RTOL)

    def test_module_cas2mach(self):
        """Test module-level cas2mach function."""
        mach = aero.cas2mach(200, 10000)
        assert mach == pytest.approx(EXPECTED["mach_from_cas"], rel=RTOL)

    def test_module_crossover_alt(self):
        """Test module-level crossover_alt function."""
        h = aero.crossover_alt(150, 0.78)
        assert h == pytest.approx(EXPECTED["crossover_alt"], rel=0.01)


class TestAeroArrayInputs:
    """Tests for array inputs."""

    def test_atmos_array(self):
        """Test atmospheric properties with array inputs."""
        aero_obj = Aero()
        h = np.array([0, 5000, 10000, 15000])
        p, rho, T = aero_obj.atmos(h)

        assert isinstance(T, np.ndarray)
        assert T.shape == (4,)
        assert T[0] == pytest.approx(EXPECTED["temperature_0m"], rel=RTOL)
        assert T[2] == pytest.approx(EXPECTED["temperature_10km"], rel=RTOL)

    def test_distance_array(self):
        """Test distance calculation with array inputs."""
        aero_obj = Aero()

        lat1 = np.array([51.5, 40.7])
        lon1 = np.array([-0.1, -74.0])
        lat2 = np.array([48.85, 34.05])
        lon2 = np.array([2.35, -118.25])

        dist = aero_obj.distance(lat1, lon1, lat2, lon2)

        assert isinstance(dist, np.ndarray)
        assert dist.shape == (2,)
        assert dist[0] == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_airspeed_conversion_array(self):
        """Test airspeed conversions with array inputs."""
        aero_obj = Aero()

        v_cas = np.array([150, 200, 250])
        h = np.array([5000, 10000, 12000])

        tas = aero_obj.cas2tas(v_cas, h)

        assert isinstance(tas, np.ndarray)
        assert tas.shape == (3,)
        # TAS should be greater than CAS at altitude
        assert np.all(tas > v_cas)


class TestAeroCasadiBackend:
    """Tests for aero functions with CasADi backend."""

    @pytest.fixture
    def casadi(self):
        """Import casadi if available."""
        return pytest.importorskip("casadi")

    def test_distance_symbolic(self, casadi):
        """Test distance calculation with symbolic inputs."""
        aero_obj = Aero(backend=CasadiBackend())

        lat1 = casadi.SX.sym("lat1")
        lon1 = casadi.SX.sym("lon1")
        lat2 = casadi.SX.sym("lat2")
        lon2 = casadi.SX.sym("lon2")

        dist = aero_obj.distance(lat1, lon1, lat2, lon2)
        assert isinstance(dist, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [lat1, lon1, lat2, lon2], [dist])
        result = float(f(51.5, -0.1, 48.85, 2.35))
        assert result == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_bearing_symbolic(self, casadi):
        """Test bearing calculation with symbolic inputs."""
        aero_obj = Aero(backend=CasadiBackend())

        lat1 = casadi.SX.sym("lat1")
        lon1 = casadi.SX.sym("lon1")
        lat2 = casadi.SX.sym("lat2")
        lon2 = casadi.SX.sym("lon2")

        brg = aero_obj.bearing(lat1, lon1, lat2, lon2)
        assert isinstance(brg, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [lat1, lon1, lat2, lon2], [brg])
        result = float(f(51.5, -0.1, 48.85, 2.35))
        assert result == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_h_isa_symbolic(self, casadi):
        """Test h_isa calculation with symbolic inputs."""
        aero_obj = Aero(backend=CasadiBackend())

        p = casadi.SX.sym("p")
        h = aero_obj.h_isa(p)
        assert isinstance(h, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [p], [h])
        result = float(f(26429.70))
        assert result == pytest.approx(10000.0, rel=0.01)

    def test_latlon_symbolic(self, casadi):
        """Test latlon calculation with symbolic inputs."""
        aero_obj = Aero(backend=CasadiBackend())

        lat1 = casadi.SX.sym("lat1")
        lon1 = casadi.SX.sym("lon1")
        d = casadi.SX.sym("d")
        brg = casadi.SX.sym("brg")

        lat2, lon2 = aero_obj.latlon(lat1, lon1, d, brg)
        assert isinstance(lat2, casadi.SX)
        assert isinstance(lon2, casadi.SX)

    def test_cas2tas_symbolic(self, casadi):
        """Test CAS to TAS with symbolic inputs."""
        aero_obj = Aero(backend=CasadiBackend())

        v_cas = casadi.SX.sym("v_cas")
        h = casadi.SX.sym("h")

        tas = aero_obj.cas2tas(v_cas, h)
        assert isinstance(tas, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [v_cas, h], [tas])
        result = float(f(200, 10000))
        assert result == pytest.approx(EXPECTED["tas_from_cas"], rel=RTOL)

    def test_crossover_alt_symbolic(self, casadi):
        """Test crossover altitude with symbolic inputs."""
        aero_obj = Aero(backend=CasadiBackend())

        v_cas = casadi.SX.sym("v_cas")
        mach = casadi.SX.sym("mach")

        h = aero_obj.crossover_alt(v_cas, mach)
        assert isinstance(h, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [v_cas, mach], [h])
        result = float(f(150, 0.78))
        assert result == pytest.approx(EXPECTED["crossover_alt"], rel=0.01)


class TestAeroJaxBackend:
    """Tests for aero functions with JAX backend."""

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
        aero_obj = Aero(backend=JaxBackend())

        dist = aero_obj.distance(
            jnp.array(51.5),
            jnp.array(-0.1),
            jnp.array(48.85),
            jnp.array(2.35),
        )
        assert float(dist) == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)

    def test_bearing_jax(self, jnp):
        """Test bearing calculation with JAX."""
        aero_obj = Aero(backend=JaxBackend())

        brg = aero_obj.bearing(
            jnp.array(51.5),
            jnp.array(-0.1),
            jnp.array(48.85),
            jnp.array(2.35),
        )
        assert float(brg) == pytest.approx(EXPECTED["bearing_london_paris"], rel=0.01)

    def test_h_isa_jax(self, jnp):
        """Test h_isa calculation with JAX."""
        aero_obj = Aero(backend=JaxBackend())

        h = aero_obj.h_isa(jnp.array(26429.70))
        assert float(h) == pytest.approx(10000.0, rel=0.01)

    def test_cas2tas_jax(self, jnp):
        """Test CAS to TAS with JAX."""
        aero_obj = Aero(backend=JaxBackend())

        tas = aero_obj.cas2tas(jnp.array(200.0), jnp.array(10000.0))
        assert float(tas) == pytest.approx(EXPECTED["tas_from_cas"], rel=RTOL)

    def test_crossover_alt_jax(self, jnp):
        """Test crossover altitude with JAX."""
        aero_obj = Aero(backend=JaxBackend())

        h = aero_obj.crossover_alt(jnp.array(150.0), jnp.array(0.78))
        assert float(h) == pytest.approx(EXPECTED["crossover_alt"], rel=0.01)

    def test_jit_aero_functions(self, jax, jnp):
        """Test JIT compilation of aero functions."""
        aero_obj = Aero(backend=JaxBackend())

        @jax.jit
        def compute_distance(lat1, lon1, lat2, lon2):
            return aero_obj.distance(lat1, lon1, lat2, lon2)

        dist = compute_distance(
            jnp.array(51.5),
            jnp.array(-0.1),
            jnp.array(48.85),
            jnp.array(2.35),
        )
        assert float(dist) == pytest.approx(EXPECTED["distance_london_paris"], rel=0.01)


class TestAeroConstants:
    """Tests for aero constants."""

    def test_module_constants(self):
        """Test that module constants are correct."""
        assert aero.kts == pytest.approx(0.514444, rel=1e-5)
        assert aero.ft == pytest.approx(0.3048, rel=1e-5)
        assert aero.fpm == pytest.approx(0.00508, rel=1e-5)
        assert aero.nm == pytest.approx(1852.0, rel=1e-5)
        assert aero.lbs == pytest.approx(0.453592, rel=1e-5)
        assert aero.g0 == pytest.approx(9.80665, rel=1e-5)
        assert aero.R == pytest.approx(287.05287, rel=1e-5)
        assert aero.p0 == pytest.approx(101325.0, rel=1e-5)
        assert aero.rho0 == pytest.approx(1.225, rel=1e-5)
        assert aero.T0 == pytest.approx(288.15, rel=1e-5)
        assert aero.gamma == pytest.approx(1.40, rel=1e-5)

    def test_class_constants(self):
        """Test that class constants match module constants."""
        assert Aero.kts == aero.kts
        assert Aero.ft == aero.ft
        assert Aero.fpm == aero.fpm
        assert Aero.nm == aero.nm
        assert Aero.g0 == aero.g0
        assert Aero.R == aero.R
        assert Aero.p0 == aero.p0
        assert Aero.rho0 == aero.rho0
        assert Aero.T0 == aero.T0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
