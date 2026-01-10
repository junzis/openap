"""Tests for the backend abstraction pattern.

This module tests that all three backends (NumPy, CasADi, JAX) work
correctly and produce consistent results.
"""

import numpy as np
import pytest

from openap import Aero, Drag, Emission, FuelFlow, Thrust
from openap.backends import CasadiBackend, JaxBackend, NumpyBackend


# Expected values computed with NumPy backend (reference)
EXPECTED = {
    "thrust_takeoff": 185981.10,  # N, tas=150kt, alt=0ft
    "thrust_climb": 72317.87,  # N, tas=280kt, alt=20000ft, roc=2000fpm
    "drag_clean": 47722.51,  # N, mass=65000kg, tas=250kt, alt=35000ft
    "fuelflow_enroute": 0.988612,  # kg/s, mass=65000kg, tas=250kt, alt=35000ft
    "fuelflow_at_thrust": 1.030994,  # kg/s, thrust=50000N
    "emission_nox": 16.2334,  # g/s, ffac=1.0kg/s, tas=250kt, alt=35000ft
    "emission_co2": 3160.0,  # g/s, ffac=1.0kg/s
    "emission_h2o": 1230.0,  # g/s, ffac=1.0kg/s
    "aero_temperature": 223.15,  # K, h=10000m
    "aero_density": 0.412604,  # kg/m³, h=10000m
    "aero_pressure": 26429.70,  # Pa, h=10000m
    "thrust_array": [185981.10, 141161.74, 96559.57],  # N
}

# Tolerance for floating point comparisons
RTOL = 1e-4  # 0.01% relative tolerance


class TestNumpyBackend:
    """Tests for NumpyBackend (default)."""

    def test_thrust_takeoff(self):
        """Test thrust calculation at takeoff."""
        thrust = Thrust("A320")
        assert type(thrust.backend).__name__ == "NumpyBackend"

        T = thrust.takeoff(tas=150, alt=0)
        assert isinstance(T, (float, np.floating))
        assert T == pytest.approx(EXPECTED["thrust_takeoff"], rel=RTOL)

    def test_thrust_climb(self):
        """Test thrust calculation during climb."""
        thrust = Thrust("A320")
        T = thrust.climb(tas=280, alt=20000, roc=2000)
        assert isinstance(T, (float, np.floating))
        assert T == pytest.approx(EXPECTED["thrust_climb"], rel=RTOL)

    def test_drag_clean(self):
        """Test drag calculation in clean configuration."""
        drag = Drag("A320")
        assert type(drag.backend).__name__ == "NumpyBackend"

        D = drag.clean(mass=65000, tas=250, alt=35000)
        assert isinstance(D, (float, np.floating))
        assert D == pytest.approx(EXPECTED["drag_clean"], rel=RTOL)

    def test_fuelflow_enroute(self):
        """Test fuel flow calculation."""
        ff = FuelFlow("A320")
        assert type(ff.backend).__name__ == "NumpyBackend"

        fuel = ff.enroute(mass=65000, tas=250, alt=35000)
        assert isinstance(fuel, (float, np.floating))
        assert fuel == pytest.approx(EXPECTED["fuelflow_enroute"], rel=RTOL)

    def test_fuelflow_at_thrust(self):
        """Test fuel flow at given thrust."""
        ff = FuelFlow("A320")
        fuel = ff.at_thrust(50000)
        assert isinstance(fuel, (float, np.floating))
        assert fuel == pytest.approx(EXPECTED["fuelflow_at_thrust"], rel=RTOL)

    def test_emission_nox(self):
        """Test NOx emission calculation."""
        em = Emission("A320")
        assert type(em.backend).__name__ == "NumpyBackend"

        nox = em.nox(ffac=1.0, tas=250, alt=35000)
        assert isinstance(nox, (float, np.floating))
        assert nox == pytest.approx(EXPECTED["emission_nox"], rel=RTOL)

    def test_emission_co2(self):
        """Test CO2 emission calculation."""
        em = Emission("A320")
        co2 = em.co2(ffac=1.0)
        assert co2 == pytest.approx(EXPECTED["emission_co2"], rel=RTOL)

    def test_emission_h2o(self):
        """Test H2O emission calculation."""
        em = Emission("A320")
        h2o = em.h2o(ffac=1.0)
        assert h2o == pytest.approx(EXPECTED["emission_h2o"], rel=RTOL)

    def test_aero_temperature(self):
        """Test temperature calculation."""
        aero = Aero()
        T = aero.temperature(10000)  # 10km
        assert isinstance(T, (float, np.floating))
        assert T == pytest.approx(EXPECTED["aero_temperature"], rel=RTOL)

    def test_aero_density(self):
        """Test density calculation."""
        aero = Aero()
        rho = aero.density(10000)  # 10km
        assert rho == pytest.approx(EXPECTED["aero_density"], rel=RTOL)

    def test_aero_pressure(self):
        """Test pressure calculation."""
        aero = Aero()
        p = aero.pressure(10000)  # 10km
        assert p == pytest.approx(EXPECTED["aero_pressure"], rel=RTOL)

    def test_array_inputs(self):
        """Test that array inputs work correctly."""
        thrust = Thrust("A320")
        tas = np.array([150, 200, 250])
        alt = np.array([0, 10000, 20000])

        T = thrust.takeoff(tas, alt)
        assert isinstance(T, np.ndarray)
        assert T.shape == (3,)
        np.testing.assert_allclose(T, EXPECTED["thrust_array"], rtol=RTOL)


class TestCasadiBackend:
    """Tests for CasadiBackend."""

    @pytest.fixture
    def casadi(self):
        """Import casadi if available."""
        casadi = pytest.importorskip("casadi")
        return casadi

    def test_thrust_symbolic(self, casadi):
        """Test thrust with symbolic inputs."""
        thrust = Thrust("A320", backend=CasadiBackend())
        assert type(thrust.backend).__name__ == "CasadiBackend"

        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        T = thrust.takeoff(tas, alt)

        assert isinstance(T, casadi.SX)

        # Evaluate at numeric values
        f = casadi.Function("f", [tas, alt], [T])
        result = float(f(150, 0))
        assert result == pytest.approx(EXPECTED["thrust_takeoff"], rel=RTOL)

    def test_drag_symbolic(self, casadi):
        """Test drag with symbolic inputs."""
        drag = Drag("A320", backend=CasadiBackend())

        mass = casadi.SX.sym("mass")
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        D = drag.clean(mass, tas, alt)

        assert isinstance(D, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [mass, tas, alt], [D])
        result = float(f(65000, 250, 35000))
        assert result == pytest.approx(EXPECTED["drag_clean"], rel=RTOL)

    def test_fuelflow_symbolic(self, casadi):
        """Test fuel flow with symbolic inputs."""
        ff = FuelFlow("A320", backend=CasadiBackend())

        mass = casadi.SX.sym("mass")
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        fuel = ff.enroute(mass, tas, alt)

        assert isinstance(fuel, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [mass, tas, alt], [fuel])
        result = float(f(65000, 250, 35000))
        assert result == pytest.approx(EXPECTED["fuelflow_enroute"], rel=RTOL)

    def test_emission_symbolic(self, casadi):
        """Test emission with symbolic inputs."""
        em = Emission("A320", backend=CasadiBackend())

        ffac = casadi.SX.sym("ffac")
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        nox = em.nox(ffac, tas, alt)

        assert isinstance(nox, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [ffac, tas, alt], [nox])
        result = float(f(1.0, 250, 35000))
        assert result == pytest.approx(EXPECTED["emission_nox"], rel=RTOL)

    def test_jacobian(self, casadi):
        """Test that Jacobian can be computed."""
        thrust = Thrust("A320", backend=CasadiBackend())

        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        T = thrust.takeoff(tas, alt)

        # Compute Jacobian
        jacobian = casadi.jacobian(T, tas)
        assert isinstance(jacobian, casadi.SX)

        # Evaluate (note: 'jac' is a reserved name in CasADi)
        jac_fn = casadi.Function("thrust_jacobian", [tas, alt], [jacobian])
        result = jac_fn(150, 0)
        assert result.shape == (1, 1)

        # dT/dtas should be negative (thrust decreases with speed at takeoff)
        assert float(result) < 0
        assert float(result) == pytest.approx(-276.19, rel=0.01)

    def test_aero_symbolic(self, casadi):
        """Test aero functions with symbolic inputs."""
        aero = Aero(backend=CasadiBackend())

        h = casadi.SX.sym("h")
        T = aero.temperature(h)
        assert isinstance(T, casadi.SX)

        # Evaluate
        f = casadi.Function("f", [h], [T])
        result = float(f(10000))
        assert result == pytest.approx(EXPECTED["aero_temperature"], rel=RTOL)

        # Test distance
        lat1 = casadi.SX.sym("lat1")
        lon1 = casadi.SX.sym("lon1")
        lat2 = casadi.SX.sym("lat2")
        lon2 = casadi.SX.sym("lon2")

        dist = aero.distance(lat1, lon1, lat2, lon2)
        assert isinstance(dist, casadi.SX)

        brg = aero.bearing(lat1, lon1, lat2, lon2)
        assert isinstance(brg, casadi.SX)


class TestJaxBackend:
    """Tests for JaxBackend."""

    @pytest.fixture
    def jax(self):
        """Import jax if available."""
        jax = pytest.importorskip("jax")
        return jax

    @pytest.fixture
    def jnp(self, jax):
        """Import jax.numpy."""
        return jax.numpy

    def test_thrust_jax(self, jnp):
        """Test thrust with JAX arrays."""
        thrust = Thrust("A320", backend=JaxBackend())
        assert type(thrust.backend).__name__ == "JaxBackend"

        T = thrust.takeoff(jnp.array(150.0), jnp.array(0.0))
        assert float(T) == pytest.approx(EXPECTED["thrust_takeoff"], rel=RTOL)

    def test_drag_jax(self, jnp):
        """Test drag with JAX arrays."""
        drag = Drag("A320", backend=JaxBackend())

        D = drag.clean(
            jnp.array(65000.0), jnp.array(250.0), jnp.array(35000.0)
        )
        assert float(D) == pytest.approx(EXPECTED["drag_clean"], rel=RTOL)

    def test_fuelflow_jax(self, jnp):
        """Test fuel flow with JAX arrays."""
        ff = FuelFlow("A320", backend=JaxBackend())

        fuel = ff.enroute(
            jnp.array(65000.0), jnp.array(250.0), jnp.array(35000.0)
        )
        assert float(fuel) == pytest.approx(EXPECTED["fuelflow_enroute"], rel=RTOL)

    def test_emission_jax(self, jnp):
        """Test emission with JAX arrays."""
        em = Emission("A320", backend=JaxBackend())

        nox = em.nox(jnp.array(1.0), jnp.array(250.0), jnp.array(35000.0))
        assert float(nox) == pytest.approx(EXPECTED["emission_nox"], rel=RTOL)

    def test_jit_compilation(self, jax, jnp):
        """Test that JIT compilation works."""
        thrust = Thrust("A320", backend=JaxBackend())

        @jax.jit
        def compute_thrust(tas, alt):
            return thrust.takeoff(tas, alt)

        # First call compiles
        result1 = compute_thrust(jnp.array(150.0), jnp.array(0.0))
        # Second call uses compiled version
        result2 = compute_thrust(jnp.array(200.0), jnp.array(0.0))

        assert float(result1) == pytest.approx(EXPECTED["thrust_takeoff"], rel=RTOL)
        # Different input should give different output
        assert float(result2) == pytest.approx(173103.59, rel=RTOL)

    def test_gradient(self, jax, jnp):
        """Test that gradients can be computed."""
        thrust = Thrust("A320", backend=JaxBackend())

        def thrust_fn(tas):
            return thrust.takeoff(tas, 0.0)

        grad_fn = jax.grad(thrust_fn)
        dT_dtas = grad_fn(150.0)

        # Gradient should match CasADi result
        assert not jnp.isnan(dT_dtas)
        assert float(dT_dtas) == pytest.approx(-276.19, rel=0.01)

    def test_aero_jax(self, jnp):
        """Test aero functions with JAX."""
        aero = Aero(backend=JaxBackend())

        T = aero.temperature(jnp.array(10000.0))
        assert float(T) == pytest.approx(EXPECTED["aero_temperature"], rel=RTOL)

        rho = aero.density(jnp.array(10000.0))
        assert float(rho) == pytest.approx(EXPECTED["aero_density"], rel=RTOL)

        p = aero.pressure(jnp.array(10000.0))
        assert float(p) == pytest.approx(EXPECTED["aero_pressure"], rel=RTOL)


class TestBackendConsistency:
    """Tests that all backends produce consistent results."""

    @pytest.fixture
    def casadi(self):
        return pytest.importorskip("casadi")

    @pytest.fixture
    def jax(self):
        return pytest.importorskip("jax")

    def test_thrust_consistency(self, casadi, jax):
        """Test that all backends give same thrust."""
        jnp = jax.numpy

        # NumPy
        thrust_np = Thrust("A320", backend=NumpyBackend())
        T_np = thrust_np.takeoff(tas=150, alt=0)

        # CasADi
        thrust_ca = Thrust("A320", backend=CasadiBackend())
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        T_ca_sym = thrust_ca.takeoff(tas, alt)
        f = casadi.Function("f", [tas, alt], [T_ca_sym])
        T_ca = float(f(150, 0))

        # JAX
        thrust_jax = Thrust("A320", backend=JaxBackend())
        T_jax = float(thrust_jax.takeoff(jnp.array(150.0), jnp.array(0.0)))

        # All should match expected value
        assert T_np == pytest.approx(EXPECTED["thrust_takeoff"], rel=RTOL)
        assert T_ca == pytest.approx(EXPECTED["thrust_takeoff"], rel=RTOL)
        assert T_jax == pytest.approx(EXPECTED["thrust_takeoff"], rel=RTOL)

    def test_drag_consistency(self, casadi, jax):
        """Test that all backends give same drag."""
        jnp = jax.numpy

        # NumPy
        drag_np = Drag("A320", backend=NumpyBackend())
        D_np = drag_np.clean(mass=65000, tas=250, alt=35000)

        # CasADi
        drag_ca = Drag("A320", backend=CasadiBackend())
        mass = casadi.SX.sym("mass")
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        D_ca_sym = drag_ca.clean(mass, tas, alt)
        f = casadi.Function("f", [mass, tas, alt], [D_ca_sym])
        D_ca = float(f(65000, 250, 35000))

        # JAX
        drag_jax = Drag("A320", backend=JaxBackend())
        D_jax = float(
            drag_jax.clean(
                jnp.array(65000.0), jnp.array(250.0), jnp.array(35000.0)
            )
        )

        # All should match expected value
        assert D_np == pytest.approx(EXPECTED["drag_clean"], rel=RTOL)
        assert D_ca == pytest.approx(EXPECTED["drag_clean"], rel=RTOL)
        assert D_jax == pytest.approx(EXPECTED["drag_clean"], rel=RTOL)

    def test_fuelflow_consistency(self, casadi, jax):
        """Test that all backends give same fuel flow."""
        jnp = jax.numpy

        # NumPy
        ff_np = FuelFlow("A320", backend=NumpyBackend())
        fuel_np = ff_np.enroute(mass=65000, tas=250, alt=35000)

        # CasADi
        ff_ca = FuelFlow("A320", backend=CasadiBackend())
        mass = casadi.SX.sym("mass")
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        fuel_ca_sym = ff_ca.enroute(mass, tas, alt)
        f = casadi.Function("f", [mass, tas, alt], [fuel_ca_sym])
        fuel_ca = float(f(65000, 250, 35000))

        # JAX
        ff_jax = FuelFlow("A320", backend=JaxBackend())
        fuel_jax = float(
            ff_jax.enroute(
                jnp.array(65000.0), jnp.array(250.0), jnp.array(35000.0)
            )
        )

        # All should match expected value
        assert fuel_np == pytest.approx(EXPECTED["fuelflow_enroute"], rel=RTOL)
        assert fuel_ca == pytest.approx(EXPECTED["fuelflow_enroute"], rel=RTOL)
        assert fuel_jax == pytest.approx(EXPECTED["fuelflow_enroute"], rel=RTOL)

    def test_emission_consistency(self, casadi, jax):
        """Test that all backends give same emissions."""
        jnp = jax.numpy

        # NumPy
        em_np = Emission("A320", backend=NumpyBackend())
        nox_np = em_np.nox(ffac=1.0, tas=250, alt=35000)

        # CasADi
        em_ca = Emission("A320", backend=CasadiBackend())
        ffac = casadi.SX.sym("ffac")
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        nox_ca_sym = em_ca.nox(ffac, tas, alt)
        f = casadi.Function("f", [ffac, tas, alt], [nox_ca_sym])
        nox_ca = float(f(1.0, 250, 35000))

        # JAX
        em_jax = Emission("A320", backend=JaxBackend())
        nox_jax = float(
            em_jax.nox(jnp.array(1.0), jnp.array(250.0), jnp.array(35000.0))
        )

        # All should match expected value
        assert nox_np == pytest.approx(EXPECTED["emission_nox"], rel=RTOL)
        assert nox_ca == pytest.approx(EXPECTED["emission_nox"], rel=RTOL)
        assert nox_jax == pytest.approx(EXPECTED["emission_nox"], rel=RTOL)

    def test_aero_consistency(self, casadi, jax):
        """Test that all backends give same aero values."""
        jnp = jax.numpy

        # NumPy
        aero_np = Aero(backend=NumpyBackend())
        T_np = aero_np.temperature(10000)
        rho_np = aero_np.density(10000)
        p_np = aero_np.pressure(10000)

        # CasADi
        aero_ca = Aero(backend=CasadiBackend())
        h = casadi.SX.sym("h")
        T_ca_sym = aero_ca.temperature(h)
        rho_ca_sym = aero_ca.density(h)
        p_ca_sym = aero_ca.pressure(h)
        f_T = casadi.Function("f", [h], [T_ca_sym])
        f_rho = casadi.Function("f", [h], [rho_ca_sym])
        f_p = casadi.Function("f", [h], [p_ca_sym])
        T_ca = float(f_T(10000))
        rho_ca = float(f_rho(10000))
        p_ca = float(f_p(10000))

        # JAX
        aero_jax = Aero(backend=JaxBackend())
        T_jax = float(aero_jax.temperature(jnp.array(10000.0)))
        rho_jax = float(aero_jax.density(jnp.array(10000.0)))
        p_jax = float(aero_jax.pressure(jnp.array(10000.0)))

        # All should match expected values
        assert T_np == pytest.approx(EXPECTED["aero_temperature"], rel=RTOL)
        assert T_ca == pytest.approx(EXPECTED["aero_temperature"], rel=RTOL)
        assert T_jax == pytest.approx(EXPECTED["aero_temperature"], rel=RTOL)

        assert rho_np == pytest.approx(EXPECTED["aero_density"], rel=RTOL)
        assert rho_ca == pytest.approx(EXPECTED["aero_density"], rel=RTOL)
        assert rho_jax == pytest.approx(EXPECTED["aero_density"], rel=RTOL)

        assert p_np == pytest.approx(EXPECTED["aero_pressure"], rel=RTOL)
        assert p_ca == pytest.approx(EXPECTED["aero_pressure"], rel=RTOL)
        assert p_jax == pytest.approx(EXPECTED["aero_pressure"], rel=RTOL)


class TestConvenienceModules:
    """Tests for the convenience modules (openap.casadi, openap.jax)."""

    def test_casadi_module(self):
        """Test openap.casadi convenience module."""
        casadi = pytest.importorskip("casadi")

        from openap.casadi import Drag, Emission, FuelFlow, Thrust, aero, prop

        # Check classes use CasadiBackend
        thrust = Thrust("A320")
        assert type(thrust.backend).__name__ == "CasadiBackend"

        # Check prop is available
        ac = prop.aircraft("A320")
        assert "mtow" in ac
        assert ac["mtow"] == pytest.approx(78000, rel=0.01)

        # Check aero works symbolically and gives correct values
        h = casadi.SX.sym("h")
        T = aero.temperature(h)
        assert isinstance(T, casadi.SX)

        f = casadi.Function("f", [h], [T])
        result = float(f(10000))
        assert result == pytest.approx(EXPECTED["aero_temperature"], rel=RTOL)

    def test_jax_module(self):
        """Test openap.jax convenience module."""
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        from openap.jax import Drag, Emission, FuelFlow, Thrust, aero

        # Check classes use JaxBackend
        thrust = Thrust("A320")
        assert type(thrust.backend).__name__ == "JaxBackend"

        # Check JIT works and gives correct values
        @jax.jit
        def compute(tas, alt):
            return thrust.takeoff(tas, alt)

        result = compute(jnp.array(150.0), jnp.array(0.0))
        assert float(result) == pytest.approx(EXPECTED["thrust_takeoff"], rel=RTOL)

        # Check aero gives correct values
        T = aero.temperature(jnp.array(10000.0))
        assert float(T) == pytest.approx(EXPECTED["aero_temperature"], rel=RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
