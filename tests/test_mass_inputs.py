"""Mass units, validation, and backend regression tests."""

import pytest

import numpy as np
from openap import Drag, FuelFlow
from openap.backends import CasadiBackend, JaxBackend
from openap.extra import mass_to_kg


@pytest.mark.parametrize("model_class", [Drag, FuelFlow])
def test_fraction_and_kg_are_equivalent(model_class):
    model = model_class("A320")
    method = model.clean if isinstance(model, Drag) else model.enroute
    for fraction in (0.87, 1.0):
        assert method(fraction, 420, 30000, vs=500) == pytest.approx(
            method(fraction * model.mtow, 420, 30000, vs=500)
        )
    mixed = np.array([0.87, model.mtow, 60000.0])
    expected = np.array([0.87 * model.mtow, model.mtow, 60000.0])
    np.testing.assert_allclose(method(mixed, 420, 30000), method(expected, 420, 30000))


@pytest.mark.parametrize(
    "mass", [0, -0.1, float("nan"), float("inf"), 78001, [0.87, 78001]]
)
@pytest.mark.parametrize("model_class", [Drag, FuelFlow])
def test_invalid_mass_raises(model_class, mass):
    model = model_class("A320")
    method = model.clean if isinstance(model, Drag) else model.enroute
    with pytest.raises(ValueError, match="mass"):
        method(mass, 420, 30000)


def test_nonclean_drag_accepts_fraction():
    model = Drag("A320")
    assert model.nonclean(0.87, 180, 3000, flap_angle=20) == pytest.approx(
        model.nonclean(0.87 * model.mtow, 180, 3000, flap_angle=20)
    )


@pytest.mark.parametrize("symbol_type", ["SX", "MX"])
def test_casadi_mass_inputs_and_derivative(symbol_type):
    ca = pytest.importorskip("casadi")
    model = FuelFlow("A320", backend=CasadiBackend())
    mass = getattr(ca, symbol_type).sym("mass")
    flow = model.enroute(mass, 420, 30000, vs=500)
    func = ca.Function("flow", [mass], [flow, ca.jacobian(flow, mass)])
    fraction_flow, fraction_grad = func(0.87)
    kg_flow, kg_grad = func(0.87 * model.mtow)
    assert float(fraction_flow) == pytest.approx(float(kg_flow))
    assert float(fraction_grad) == pytest.approx(float(kg_grad) * model.mtow)
    with pytest.raises(ValueError, match="MTOW"):
        model.enroute(ca.DM(model.mtow + 1), 420, 30000)


def test_jax_jit_and_concrete_mass_validation():
    jax = pytest.importorskip("jax")
    backend = JaxBackend()
    convert = jax.jit(lambda mass: mass_to_kg(mass, 78000, backend=backend))
    np.testing.assert_allclose(
        convert(jax.numpy.array([0.5, 1.0, 60000.0])), [39000, 78000, 60000]
    )
    with pytest.raises(ValueError, match="MTOW"):
        mass_to_kg(jax.numpy.array(78001), 78000, backend=backend)
