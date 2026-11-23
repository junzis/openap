"""Drag computed with the Aerox geometry drag model."""

import pytest

import numpy as np
from openap import Drag


def test_clean_drag_matches_aerox():
    from aerox import aero, load_aircraft
    from aerox.models.drag import build_drag_model

    drag = Drag("A320")
    mass, tas, alt = 65000, 450, 35000
    v, h = tas * 0.514444, alt * 0.3048
    rho = aero.density(h)
    q_s = 0.5 * rho * v**2 * drag.wing_area
    cl = mass * 9.80665 / q_s
    mach = aero.tas2mach(v, h)
    model = build_drag_model(load_aircraft("A320"))
    expected = model.cd(cl=cl, mach=mach, h=h) * q_s
    assert drag.clean(mass=mass, tas=tas, alt=alt) == pytest.approx(expected, rel=1e-3)


def test_wave_drag_can_be_switched_off():
    fast = dict(mass=65000, tas=470, alt=35000)  # about Mach 0.82
    assert Drag("A320").wave_drag
    assert Drag("A320").clean(**fast) > Drag("A320", wave_drag=False).clean(**fast)
    slow = dict(mass=65000, tas=250, alt=35000)
    assert Drag("A320").clean(**slow) == pytest.approx(
        Drag("A320", wave_drag=False).clean(**slow)
    )


def test_aircraft_only_in_aerox():
    drag = Drag("C172")
    assert drag.aircraft["mtow"] > 0
    assert drag.clean(mass=1100, tas=110, alt=5000) > 0
    assert drag.nonclean(mass=1100, tas=60, alt=500, flap_angle=20) > 0


def test_nonclean_without_flaps_equals_clean():
    drag = Drag("A320")
    args = dict(mass=60000, tas=150, alt=1000)
    clean = drag.clean(**args)
    assert drag.nonclean(**args, flap_angle=0) == pytest.approx(clean)
    assert drag.nonclean(**args, flap_angle=20, landing_gear=True) > drag.nonclean(
        **args, flap_angle=20
    )


def test_nonclean_matches_aerox():
    from aerox import aero, load_aircraft
    from aerox.models.drag import DragCondition, build_drag_model

    drag = Drag("A320")
    mass, tas, alt = 60000, 150, 1000
    v, h = tas * 0.514444, alt * 0.3048
    q_s = 0.5 * aero.density(h) * v**2 * drag.wing_area
    cl = mass * 9.80665 / q_s
    mach = aero.tas2mach(v, h)
    condition = DragCondition(
        mach,
        aero.reynolds_per_m(mach, h),
        flap_angle_rad=np.deg2rad(30),
        gear_extension_fraction=1.0,
    )
    model = build_drag_model(load_aircraft("A320"))
    expected = model.cd(cl=cl, condition=condition) * q_s
    actual = drag.nonclean(
        mass=mass, tas=tas, alt=alt, flap_angle=30, landing_gear=True
    )
    assert actual == pytest.approx(expected, rel=1e-3)


def test_flaps_add_drag():
    drag = Drag("A320")
    args = dict(mass=60000, tas=150, alt=1000)
    values = [drag.nonclean(**args, flap_angle=f) for f in (0, 10, 20, 30, 40)]
    assert values == sorted(values)


def test_nonclean_with_casadi_and_jax():
    from openap.backends import CasadiBackend, JaxBackend

    args = dict(mass=60000, tas=150, alt=1000, flap_angle=30, landing_gear=True)
    expected = Drag("A320").nonclean(**args)

    casadi = pytest.importorskip("casadi")
    flap = casadi.SX.sym("flap")
    drag = Drag("A320", backend=CasadiBackend())
    symbolic = drag.nonclean(**{**args, "flap_angle": flap})
    value = float(casadi.Function("d", [flap], [symbolic])(30))
    # CasADi uses smooth maxima, so allow a small difference.
    assert value == pytest.approx(expected, rel=1e-3)

    pytest.importorskip("jax")
    drag = Drag("A320", backend=JaxBackend())
    value = float(np.ravel(drag.nonclean(**args))[0])
    assert value == pytest.approx(expected, rel=1e-5)


def test_drag_accepts_arrays():
    drag = Drag("B738")
    result = drag.clean(mass=[60000, 65000], tas=[250, 450], alt=[10000, 35000])
    assert np.shape(result) == (2,)
    assert np.all(np.isfinite(result))


def test_unknown_aircraft_raises():
    with pytest.raises(ValueError, match="not available"):
        Drag("ZZZZ")
