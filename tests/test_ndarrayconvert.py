"""Tests for the ndarrayconvert decorator."""

import numpy as np
import pytest

from openap.extra import ndarrayconvert


class MockModel:
    """Mock model class for testing the decorator."""

    @ndarrayconvert
    def compute(self, x, y):
        """Simple computation that requires array inputs."""
        return x + y

    @ndarrayconvert
    def compute_multi(self, x, y):
        """Return multiple values."""
        return x + y, x * y

    @ndarrayconvert(column=True)
    def compute_column(self, x, y):
        """Computation with column reshaping."""
        return x + y


class TestNdarrayConvertNumPy:
    """Test ndarrayconvert with NumPy inputs."""

    def test_scalar_input_returns_scalar(self):
        """Scalar inputs should return scalar output."""
        model = MockModel()
        result = model.compute(1.0, 2.0)
        assert isinstance(result, float)
        assert result == 3.0

    def test_int_input_returns_scalar(self):
        """Integer inputs should return scalar output."""
        model = MockModel()
        result = model.compute(1, 2)
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert result == 3

    def test_list_input_returns_array(self):
        """List inputs should return array output."""
        model = MockModel()
        result = model.compute([1, 2, 3], [4, 5, 6])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5, 7, 9])

    def test_array_input_returns_array(self):
        """Array inputs should return array output."""
        model = MockModel()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        result = model.compute(x, y)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5.0, 7.0, 9.0])

    def test_single_element_array_returns_scalar(self):
        """Single-element array should return scalar."""
        model = MockModel()
        result = model.compute(np.array([1.0]), np.array([2.0]))
        assert isinstance(result, float)
        assert result == 3.0

    def test_multi_return_scalar(self):
        """Multiple return values with scalar inputs."""
        model = MockModel()
        sum_result, prod_result = model.compute_multi(2.0, 3.0)
        assert isinstance(sum_result, float)
        assert isinstance(prod_result, float)
        assert sum_result == 5.0
        assert prod_result == 6.0

    def test_multi_return_array(self):
        """Multiple return values with array inputs."""
        model = MockModel()
        sum_result, prod_result = model.compute_multi([1, 2], [3, 4])
        assert isinstance(sum_result, np.ndarray)
        assert isinstance(prod_result, np.ndarray)
        np.testing.assert_array_equal(sum_result, [4, 6])
        np.testing.assert_array_equal(prod_result, [3, 8])

    def test_column_mode(self):
        """Test column=True reshapes to column vectors."""
        model = MockModel()
        result = model.compute_column([1, 2, 3], [4, 5, 6])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)

    def test_kwargs(self):
        """Test with keyword arguments."""
        model = MockModel()
        result = model.compute(x=1.0, y=2.0)
        assert isinstance(result, float)
        assert result == 3.0


class TestNdarrayConvertCasadi:
    """Test ndarrayconvert with CasADi inputs."""

    @pytest.fixture
    def casadi(self):
        return pytest.importorskip("casadi")

    def test_symbolic_passthrough(self, casadi):
        """CasADi symbolic types should pass through unchanged."""
        model = MockModel()
        x = casadi.SX.sym("x")
        y = casadi.SX.sym("y")
        result = model.compute(x, y)
        assert isinstance(result, casadi.SX)

    def test_mx_passthrough(self, casadi):
        """CasADi MX types should pass through unchanged."""
        model = MockModel()
        x = casadi.MX.sym("x")
        y = casadi.MX.sym("y")
        result = model.compute(x, y)
        assert isinstance(result, casadi.MX)

    def test_dm_passthrough(self, casadi):
        """CasADi DM types should pass through unchanged."""
        model = MockModel()
        x = casadi.DM([1, 2, 3])
        y = casadi.DM([4, 5, 6])
        result = model.compute(x, y)
        assert isinstance(result, casadi.DM)


class TestNdarrayConvertJax:
    """Test ndarrayconvert with JAX inputs."""

    @pytest.fixture
    def jax(self):
        return pytest.importorskip("jax")

    @pytest.fixture
    def jnp(self, jax):
        return jax.numpy

    def test_jax_array_passthrough(self, jnp):
        """JAX arrays should pass through unchanged."""
        model = MockModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        result = model.compute(x, y)
        # Result should be JAX array, not converted to NumPy
        assert "jax" in type(result).__module__

    def test_jax_scalar_passthrough(self, jnp):
        """JAX scalars should pass through unchanged."""
        model = MockModel()
        x = jnp.array(1.0)
        y = jnp.array(2.0)
        result = model.compute(x, y)
        assert "jax" in type(result).__module__


class TestNdarrayConvertIntegration:
    """Integration tests with actual OpenAP models."""

    def test_thrust_scalar(self):
        """Thrust model with scalar inputs."""
        from openap import Thrust

        thrust = Thrust("A320")
        result = thrust.takeoff(tas=150, alt=0)
        assert isinstance(result, float)
        assert result > 0

    def test_thrust_array(self):
        """Thrust model with array inputs."""
        from openap import Thrust

        thrust = Thrust("A320")
        result = thrust.takeoff(tas=[150, 200, 250], alt=[0, 0, 0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert all(r > 0 for r in result)

    def test_drag_scalar(self):
        """Drag model with scalar inputs."""
        from openap import Drag

        drag = Drag("A320")
        result = drag.clean(mass=65000, tas=250, alt=35000)
        assert isinstance(result, float)
        assert result > 0

    def test_fuelflow_scalar(self):
        """FuelFlow model with scalar inputs."""
        from openap import FuelFlow

        ff = FuelFlow("A320")
        result = ff.enroute(mass=65000, tas=250, alt=35000)
        assert isinstance(result, float)
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
