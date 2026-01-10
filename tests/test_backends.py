"""Tests for the backend abstraction pattern.

This module tests that all three backends (NumPy, CasADi, JAX) work
correctly and produce consistent results.
"""

import numpy as np
import pytest

from openap import Aero, Drag, Emission, FuelFlow, Thrust
from openap.backends import CasadiBackend, JaxBackend, NumpyBackend


class TestNumpyBackend:
    """Tests for NumpyBackend (default)."""

    def test_thrust_takeoff(self):
        """Test thrust calculation at takeoff."""
        thrust = Thrust("A320")
        assert type(thrust.backend).__name__ == "NumpyBackend"

        T = thrust.takeoff(tas=150, alt=0)
        assert isinstance(T, (float, np.floating))
        assert T > 0
        assert T < 300000  # Reasonable upper bound

    def test_thrust_climb(self):
        """Test thrust calculation during climb."""
        thrust = Thrust("A320")
        T = thrust.climb(tas=280, alt=20000, roc=2000)
        assert isinstance(T, (float, np.floating))
        assert T > 0

    def test_drag_clean(self):
        """Test drag calculation in clean configuration."""
        drag = Drag("A320")
        assert type(drag.backend).__name__ == "NumpyBackend"

        D = drag.clean(mass=65000, tas=250, alt=35000)
        assert isinstance(D, (float, np.floating))
        assert D > 0
        assert D < 100000  # Reasonable upper bound

    def test_fuelflow_enroute(self):
        """Test fuel flow calculation."""
        ff = FuelFlow("A320")
        assert type(ff.backend).__name__ == "NumpyBackend"

        fuel = ff.enroute(mass=65000, tas=250, alt=35000)
        assert isinstance(fuel, (float, np.floating))
        assert fuel > 0
        assert fuel < 5  # kg/s, reasonable upper bound

    def test_emission_nox(self):
        """Test NOx emission calculation."""
        em = Emission("A320")
        assert type(em.backend).__name__ == "NumpyBackend"

        nox = em.nox(ffac=1.0, tas=250, alt=35000)
        assert isinstance(nox, (float, np.floating))
        assert nox > 0

    def test_aero_temperature(self):
        """Test temperature calculation."""
        aero = Aero()
        T = aero.temperature(10000)  # 10km
        assert isinstance(T, (float, np.floating))
        assert 200 < T < 290  # K, reasonable range

    def test_array_inputs(self):
        """Test that array inputs work correctly."""
        thrust = Thrust("A320")
        tas = np.array([150, 200, 250])
        alt = np.array([0, 10000, 20000])

        T = thrust.takeoff(tas, alt)
        assert isinstance(T, np.ndarray)
        assert T.shape == (3,)
        assert all(T > 0)


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
        result = f(150, 0)
        assert float(result) > 0

    def test_drag_symbolic(self, casadi):
        """Test drag with symbolic inputs."""
        drag = Drag("A320", backend=CasadiBackend())

        mass = casadi.SX.sym("mass")
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        D = drag.clean(mass, tas, alt)

        assert isinstance(D, casadi.SX)

    def test_fuelflow_symbolic(self, casadi):
        """Test fuel flow with symbolic inputs."""
        ff = FuelFlow("A320", backend=CasadiBackend())

        mass = casadi.SX.sym("mass")
        tas = casadi.SX.sym("tas")
        alt = casadi.SX.sym("alt")
        fuel = ff.enroute(mass, tas, alt)

        assert isinstance(fuel, casadi.SX)

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

    def test_aero_symbolic(self, casadi):
        """Test aero functions with symbolic inputs."""
        aero = Aero(backend=CasadiBackend())

        h = casadi.SX.sym("h")
        T = aero.temperature(h)
        assert isinstance(T, casadi.SX)

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
        assert float(T) > 0

    def test_drag_jax(self, jnp):
        """Test drag with JAX arrays."""
        drag = Drag("A320", backend=JaxBackend())

        D = drag.clean(
            jnp.array(65000.0), jnp.array(250.0), jnp.array(35000.0)
        )
        assert float(D) > 0

    def test_fuelflow_jax(self, jnp):
        """Test fuel flow with JAX arrays."""
        ff = FuelFlow("A320", backend=JaxBackend())

        fuel = ff.enroute(
            jnp.array(65000.0), jnp.array(250.0), jnp.array(35000.0)
        )
        assert float(fuel) > 0

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

        assert float(result1) > 0
        assert float(result2) > 0
        assert float(result1) != float(result2)

    def test_gradient(self, jax, jnp):
        """Test that gradients can be computed."""
        thrust = Thrust("A320", backend=JaxBackend())

        def thrust_fn(tas):
            return thrust.takeoff(tas, 0.0)

        grad_fn = jax.grad(thrust_fn)
        dT_dtas = grad_fn(150.0)

        # Gradient should exist and be non-zero
        assert not jnp.isnan(dT_dtas)

    def test_aero_jax(self, jnp):
        """Test aero functions with JAX."""
        aero = Aero(backend=JaxBackend())

        T = aero.temperature(jnp.array(10000.0))
        assert 200 < float(T) < 290


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

        # All should be close
        assert abs(T_np - T_ca) / T_np < 0.001  # <0.1% difference
        assert abs(T_np - T_jax) / T_np < 0.001

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

        # All should be close
        assert abs(D_np - D_ca) / D_np < 0.001
        assert abs(D_np - D_jax) / D_np < 0.001

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

        # All should be close
        assert abs(fuel_np - fuel_ca) / fuel_np < 0.001
        assert abs(fuel_np - fuel_jax) / fuel_np < 0.001


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

        # Check aero works symbolically
        h = casadi.SX.sym("h")
        T = aero.temperature(h)
        assert isinstance(T, casadi.SX)

    def test_jax_module(self):
        """Test openap.jax convenience module."""
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        from openap.jax import Drag, Emission, FuelFlow, Thrust, aero

        # Check classes use JaxBackend
        thrust = Thrust("A320")
        assert type(thrust.backend).__name__ == "JaxBackend"

        # Check JIT works
        @jax.jit
        def compute(tas, alt):
            return thrust.takeoff(tas, alt)

        result = compute(jnp.array(150.0), jnp.array(0.0))
        assert float(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
