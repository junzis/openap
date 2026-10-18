from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

import pytest

import numpy as np
from openap.addon import bada3, bada4
from openap.backends import CasadiBackend


@pytest.fixture
def casadi():
    return pytest.importorskip("casadi")


@pytest.fixture
def bada3_model():
    return {
        "wing": {"area": 122.6},
        "engine": {"type": "turbofan", "number": 2},
        "CD0": {"CR": 0.02, "IC": 0.025, "TO": 0.03, "AP": 0.035, "LD": 0.04},
        "CD2": {"CR": 0.04, "IC": 0.045, "TO": 0.05, "AP": 0.055, "LD": 0.06},
        "CD0_lgear": 0.02,
        "Ct": [145000.0, 50000.0, 1e-10, 0.0, 0.01],
        "CTdeshigh": 0.15,
        "CTdeslow": 0.08,
        "CTdesapp": 0.10,
        "CTdesld": 0.12,
        "HpDes": 8000.0,
        "Cf": [0.6, 1200.0],
        "CfDes": [300.0, 80000.0],
        "CfCrz": 0.95,
    }


@pytest.fixture
def bada4_path(tmp_path):
    model_dir = tmp_path / "A320-TEST"
    model_dir.mkdir()

    def tags(name, values):
        return "".join(f"<{name}>{value}</{name}>" for value in values)

    d_values = [
        0.018,
        0.002,
        -0.0004,
        0.00008,
        -0.00001,
        0.04,
        0.003,
        -0.0005,
        0.00007,
        -0.00001,
        0.001,
        0.0002,
        -0.00003,
        0.000004,
        -0.0000005,
    ]
    a_values = [0.0] * 36
    a_values[0] = 0.2
    a_values[1] = 0.03
    a_values[6] = 0.02
    ti_values = [0.0] * 12
    ti_values[0] = 0.04
    ti_values[1] = 0.01
    ti_values[4] = 0.005
    b_values = [0.0] * 36
    b_values[0] = 0.9
    b_values[1] = 0.02
    b_values[6] = 0.03
    b_takeoff_values = [0.0] * 36
    b_takeoff_values[0] = 1.4
    b_takeoff_values[1] = 0.05
    b_takeoff_values[6] = 0.04
    c_values = [0.0] * 45
    c_values[0] = 1.0
    c_values[1] = 0.01
    c_values[5] = 0.02
    f_values = [0.0] * 25
    f_values[0] = 0.08
    f_values[1] = 0.01
    f_values[5] = 0.02
    fi_values = [0.0] * 9
    fi_values[0] = 0.03
    fi_values[1] = 0.004
    fi_values[3] = 0.002

    xml = f"""
    <root>
      <AFCM><S>122.6</S></AFCM>
      <PFM>
        <MREF>60000.0</MREF>
        <LHV>43000000.0</LHV>
        <TFM>
          <CT>{tags("a", a_values)}</CT>
          <LIDL>
            <CT>{tags("ti", ti_values)}</CT>
            <CF>{tags("fi", fi_values)}</CF>
          </LIDL>
          <MCRZ>
            <kink>10.0</kink>
            <flat_rating>{tags("b", b_values)}</flat_rating>
            <temp_rating>{tags("c", c_values)}</temp_rating>
          </MCRZ>
          <MCMB>
            <kink>10.0</kink>
            <flat_rating>{tags("b", b_values)}</flat_rating>
            <temp_rating>{tags("c", c_values)}</temp_rating>
          </MCMB>
          <MTKF>
            <kink>10.0</kink>
            <flat_rating>{tags("b", b_takeoff_values)}</flat_rating>
            <temp_rating>{tags("c", c_values)}</temp_rating>
          </MTKF>
          <CF>{tags("f", f_values)}</CF>
        </TFM>
      </PFM>
      <PERF>
        <DPM_clean>
          <scalar>1.0</scalar>
          <M_max>0.82</M_max>
        </DPM_clean>
        <CD_clean>{tags("d", d_values)}</CD_clean>
      </PERF>
    </root>
    """
    (model_dir / "model.xml").write_text(xml, encoding="utf-8")
    return str(tmp_path)


def _eval1(casadi, expr, *symbols_and_values):
    symbols = [symbol for symbol, _ in symbols_and_values]
    values = [value for _, value in symbols_and_values]
    fn = casadi.Function("f", symbols, [expr])
    return float(fn(*values))


def _poly2(coefficients, shape, row_terms, col_terms):
    rows, cols = shape
    total = 0.0
    for i in range(rows):
        for j in range(cols):
            total += coefficients[i * cols + j] * row_terms[i] * col_terms[j]
    return total


def _manual_bada4_rating_thrust(ac, bada_path, tas, alt, rating, dT=0.0):
    thrust = bada4.Thrust(ac, bada_path)
    bxml = bada4.load_bada4(ac, bada_path)
    a_coeff = [float(v.text) for v in bxml.findall("./PFM/TFM/CT/a")]
    kink = float(bxml.findtext(f"./PFM/TFM/{rating}/kink"))
    b_coeff = [float(v.text) for v in bxml.findall(f"./PFM/TFM/{rating}/flat_rating/b")]
    c_coeff = [float(v.text) for v in bxml.findall(f"./PFM/TFM/{rating}/temp_rating/c")]

    h = alt * thrust.aero.ft
    v = tas * thrust.aero.kts
    mach = thrust.aero.tas2mach(v, h, dT=dT)
    delta = thrust.aero.pressure(h, dT=dT) / thrust.aero.p0
    theta = thrust.aero.temperature(h, dT=dT) / thrust.aero.T0

    mach_terms_6 = [mach**i for i in range(6)]
    if dT <= kink:
        delta_terms_6 = [delta**i for i in range(6)]
        delta_t = _poly2(b_coeff, (6, 6), delta_terms_6, mach_terms_6)
    else:
        theta_t = theta * (1 + mach**2 * (1.4 - 1) / 2)
        temp_terms = [theta_t**i for i in range(5)] + [delta**i for i in range(1, 5)]
        mach_terms_5 = [mach**i for i in range(5)]
        delta_t = _poly2(c_coeff, (9, 5), temp_terms, mach_terms_5)

    delta_t_terms = [delta_t**i for i in range(6)]
    cT = _poly2(a_coeff, (6, 6), delta_t_terms, mach_terms_6)

    return delta * thrust.m_ref * thrust.aero.g0 * cT


def test_bada3_drag_clean_supports_casadi_symbolics(casadi, bada3_model):
    numeric = bada3.Drag("A320", bada_path="", model=bada3_model).clean(
        mass=60000.0, tas=300.0, alt=12000.0
    )

    drag = bada3.Drag("A320", bada_path="", model=bada3_model, backend=CasadiBackend())
    mass = casadi.SX.sym("mass")
    tas = casadi.SX.sym("tas")
    alt = casadi.SX.sym("alt")

    result = _eval1(
        casadi,
        drag.clean(mass, tas, alt),
        (mass, 60000),
        (tas, 300),
        (alt, 12000),
    )

    assert result == pytest.approx(numeric)


def test_bada3_thrust_idle_supports_symbolic_altitude_branch(casadi, bada3_model):
    numeric = bada3.Thrust("A320", bada_path="", model=bada3_model).idle(
        tas=280.0, alt=12000.0, config="CR"
    )

    thrust = bada3.Thrust(
        "A320", bada_path="", model=bada3_model, backend=CasadiBackend()
    )
    tas = casadi.SX.sym("tas")
    alt = casadi.SX.sym("alt")

    result = _eval1(
        casadi,
        thrust.idle(tas, alt, config="CR"),
        (tas, 280),
        (alt, 12000),
    )

    assert result == pytest.approx(numeric)


def test_bada3_thrust_idle_matches_original_numeric_branches(bada3_model):
    thrust = bada3.Thrust("A320", bada_path="", model=bada3_model)
    tas = 280.0
    dT = 0.0

    for alt, config, ctdes in [
        (12000.0, "CR", bada3_model["CTdeslow"]),
        (4000.0, "AP", bada3_model["CTdesapp"]),
        (4000.0, "LD", bada3_model["CTdesld"]),
    ]:
        thr_cl_max = thrust.climb(tas, alt, dT)
        if alt > bada3_model["HpDes"]:
            expected = bada3_model["CTdeshigh"] * thr_cl_max
        else:
            expected = ctdes * thr_cl_max

        assert thrust.idle(tas, alt, dT=dT, config=config) == pytest.approx(expected)


def test_bada3_hpdes_floor_applies_only_when_nonclean_data_available(bada3_model):
    model_with_nonclean = deepcopy(bada3_model)
    model_with_nonclean["HpDes"] = 3900.0
    with_nonclean = bada3.Thrust("A320", bada_path="", model=model_with_nonclean)

    model_without_nonclean = deepcopy(model_with_nonclean)
    model_without_nonclean["CD0"]["AP"] = 0.0
    model_without_nonclean["CD0"]["LD"] = 0.0
    model_without_nonclean["CD0_lgear"] = 0.0
    model_without_nonclean["CD2"]["AP"] = 0.0
    model_without_nonclean["CD2"]["LD"] = 0.0
    without_nonclean = bada3.Thrust("A320", bada_path="", model=model_without_nonclean)

    assert with_nonclean.hpdes == pytest.approx(8000.0)
    assert without_nonclean.hpdes == pytest.approx(3900.0)


def test_bada3_fuel_enroute_forwards_backend_to_nested_models(casadi, bada3_model):
    numeric = bada3.FuelFlow("A320", bada_path="", model=bada3_model).enroute(
        mass=60000.0, tas=300.0, alt=12000.0, vs=500.0
    )

    fuel_flow = bada3.FuelFlow(
        "A320", bada_path="", model=bada3_model, backend=CasadiBackend()
    )
    mass = casadi.SX.sym("mass")
    tas = casadi.SX.sym("tas")
    alt = casadi.SX.sym("alt")
    vs = casadi.SX.sym("vs")

    result = _eval1(
        casadi,
        fuel_flow.enroute(mass, tas, alt, vs),
        (mass, 60000),
        (tas, 300),
        (alt, 12000),
        (vs, 500),
    )

    assert result == pytest.approx(numeric)


def test_bada3_fuel_enroute_does_not_apply_cruise_correction(bada3_model):
    fuel_flow = bada3.FuelFlow("A320", bada_path="", model=bada3_model)

    nominal = fuel_flow.nominal(mass=60000.0, tas=300.0, alt=12000.0, vs=500.0)
    enroute = fuel_flow.enroute(mass=60000.0, tas=300.0, alt=12000.0, vs=500.0)
    cruise = fuel_flow.cruise(mass=60000.0, tas=300.0, alt=12000.0, vs=500.0)

    assert enroute == pytest.approx(nominal)
    assert cruise == pytest.approx(bada3_model["CfCrz"] * nominal)


def test_bada3_drag_uses_temperature_deviation_in_density(bada3_model):
    drag = bada3.Drag("A320", bada_path="", model=bada3_model)

    result = drag.clean(mass=60000.0, tas=300.0, alt=12000.0, dT=10.0)

    h = 12000.0 * drag.aero.ft
    v = 300.0 * drag.aero.kts
    rho = drag.aero.density(h, dT=10.0)
    qS = 0.5 * rho * v**2 * bada3_model["wing"]["area"]
    cl = 60000.0 * drag.aero.g0 / max(qS, 1e-3)
    expected = (bada3_model["CD0"]["CR"] + bada3_model["CD2"]["CR"] * cl**2) * qS

    assert result == pytest.approx(expected)
    assert result != pytest.approx(
        drag.clean(mass=60000.0, tas=300.0, alt=12000.0, dT=0.0)
    )


def test_bada3_fuel_nominal_uses_temperature_deviation_in_thrust_cap(bada3_model):
    # Keep the idle floor below nominal fuel so this tests the thrust cap.
    bada3_model["CfDes"] = [10.0, 80000.0]
    fuel_flow = bada3.FuelFlow("A320", bada_path="", model=bada3_model)

    result = fuel_flow.nominal(mass=60000.0, tas=300.0, alt=12000.0, vs=6000.0, dT=20.0)

    eta = bada3_model["Cf"][0] * (1 + 300.0 / bada3_model["Cf"][1]) * 1e-3
    expected = eta * fuel_flow.thrust.climb(tas=300.0, alt=12000.0, dT=20.0)
    expected = max(expected, 60.0 * fuel_flow.idle(60000.0, 300.0, 12000.0, 6000.0))

    assert result == pytest.approx(expected / 60.0)
    assert result < fuel_flow.nominal(
        mass=60000.0, tas=300.0, alt=12000.0, vs=6000.0, dT=0.0
    )


def test_bada3_clean_drag_polar_params_expose_quadratic_coefficients(bada3_model):
    drag = bada3.Drag("A320", bada_path="", model=bada3_model)

    params = drag.clean_drag_polar_params(tas=300.0, alt=12000.0)

    assert params["cd0"] == pytest.approx(bada3_model["CD0"]["CR"])
    assert params["cd2"] == pytest.approx(bada3_model["CD2"]["CR"])
    assert params["cd6"] == pytest.approx(0.0)


def test_bada4_drag_clean_supports_casadi_symbolics(casadi, bada4_path):
    numeric = bada4.Drag("A320-TEST", bada4_path).clean(
        mass=60000.0, tas=300.0, alt=12000.0
    )

    drag = bada4.Drag("A320-TEST", bada4_path, backend=CasadiBackend())
    mass = casadi.SX.sym("mass")
    tas = casadi.SX.sym("tas")
    alt = casadi.SX.sym("alt")

    result = _eval1(
        casadi,
        drag.clean(mass, tas, alt),
        (mass, 60000),
        (tas, 300),
        (alt, 12000),
    )

    assert result == pytest.approx(numeric)


def test_bada4_thrust_climb_supports_casadi_symbolics(casadi, bada4_path):
    numeric = bada4.Thrust("A320-TEST", bada4_path).climb(tas=300.0, alt=12000.0)

    thrust = bada4.Thrust("A320-TEST", bada4_path, backend=CasadiBackend())
    tas = casadi.SX.sym("tas")
    alt = casadi.SX.sym("alt")

    result = _eval1(
        casadi,
        thrust.climb(tas, alt),
        (tas, 300),
        (alt, 12000),
    )

    assert result == pytest.approx(numeric)


def test_bada4_thrust_uses_temperature_deviation_in_atmosphere(bada4_path):
    thrust = bada4.Thrust("A320-TEST", bada4_path)

    result = thrust.climb(tas=300.0, alt=12000.0, dT=5.0)
    expected = _manual_bada4_rating_thrust(
        "A320-TEST", bada4_path, 300.0, 12000.0, "MCMB", dT=5.0
    )

    assert result == pytest.approx(expected)
    assert result != pytest.approx(thrust.climb(tas=300.0, alt=12000.0, dT=0.0))


def test_bada4_takeoff_uses_mtkf_rating(bada4_path):
    thrust = bada4.Thrust("A320-TEST", bada4_path)

    result = thrust.takeoff(tas=150.0, alt=0.0)
    expected = _manual_bada4_rating_thrust("A320-TEST", bada4_path, 150.0, 0.0, "MTKF")

    assert result == pytest.approx(expected)
    assert result != pytest.approx(thrust.climb(tas=150.0, alt=0.0))


def test_bada4_takeoff_falls_back_to_climb_when_mtkf_missing(bada4_path):
    xml_path = Path(bada4_path) / "A320-TEST" / "model.xml"
    tree = ElementTree.parse(xml_path)
    tfm = tree.find("./PFM/TFM")
    mtkf = tfm.find("./MTKF")
    tfm.remove(mtkf)
    tree.write(xml_path, encoding="utf-8")

    thrust = bada4.Thrust("A320-TEST", bada4_path)

    assert thrust.takeoff(tas=150.0, alt=0.0) == pytest.approx(
        thrust.climb(tas=150.0, alt=0.0)
    )


def test_bada4_thrust_polynomial_matches_original_einsum_formula(bada4_path):
    thrust = bada4.Thrust("A320-TEST", bada4_path)
    mach = np.array([[0.45]])
    h = np.array([[12000.0 * thrust.aero.ft]])

    delta = thrust.aero.pressure(h) / thrust.aero.p0

    b_matrix = np.reshape(thrust.b_["MCMB"], (6, 6))
    mach_pow = np.array([mach**i for i in range(6)]).reshape(6, -1)
    ratio_pow = np.array([delta**j for j in range(6)]).reshape(6, -1)
    delta_t = np.einsum("ij,jk,ik->k", b_matrix, mach_pow, ratio_pow)

    a_matrix = np.reshape(thrust.a_, (6, 6))
    mach_pow = np.array([mach**i for i in range(6)]).reshape(6, -1)
    delta_t_pow = np.array([delta_t**j for j in range(6)]).reshape(6, -1)
    expected = np.einsum("ij,jk,ik->k", a_matrix, mach_pow, delta_t_pow)

    result = thrust.cT(mach, h, "MCMB", dT=0.0)

    assert result == pytest.approx(float(expected[0]))


def test_bada4_drag_polynomial_matches_original_dot_formula(bada4_path):
    drag = bada4.Drag("A320-TEST", bada4_path)
    mass = np.array([[60000.0]])
    tas = np.array([[300.0]])
    alt = np.array([[12000.0]])

    v = tas * drag.aero.kts
    h = alt * drag.aero.ft
    mach = drag.aero.tas2mach(v, h)
    rho = drag.aero.density(h)
    qS = 0.5 * rho * v**2 * drag.S
    cl = mass * drag.aero.g0 / np.maximum(qS, 1e-3)
    mm = (1 - mach**2) ** (-0.5)

    d = np.array(drag.d_)
    c0 = np.dot(np.array([mm[:, 0] ** i for i in range(5)]).T, d[0:5].reshape(5, 1))
    c2 = np.dot(
        np.array([mm[:, 0] ** i for i in range(0, 13, 3)]).T,
        d[5:10].reshape(5, 1),
    )
    c6 = d[10] + np.dot(
        np.array([mm[:, 0] ** i for i in range(14, 18)]).T,
        d[11:15].reshape(4, 1),
    )
    expected_cd = drag.scalar * (c0 + c2 * cl**2 + c6 * cl**6)
    expected_drag = expected_cd * qS

    assert drag.clean(mass, tas, alt) == pytest.approx(expected_drag)


def test_bada4_clean_drag_polar_params_reconstruct_drag(bada4_path):
    drag = bada4.Drag("A320-TEST", bada4_path)
    mass = 60000.0
    tas = 300.0
    alt = 12000.0

    params = drag.clean_drag_polar_params(tas=tas, alt=alt)

    h = alt * drag.aero.ft
    v = tas * drag.aero.kts
    rho = drag.aero.density(h)
    qS = 0.5 * rho * v**2 * drag.S
    cl = mass * drag.aero.g0 / max(qS, 1e-3)
    expected = qS * (params["cd0"] + params["cd2"] * cl**2 + params["cd6"] * cl**6)

    assert drag.clean(mass=mass, tas=tas, alt=alt) == pytest.approx(expected)


def test_bada4_clean_drag_polar_params_reconstruct_divergence_drag(bada4_path):
    drag = bada4.Drag("A320-TEST", bada4_path)
    mass = 60000.0
    alt = 12000.0
    h = alt * drag.aero.ft
    tas = drag.aero.mach2tas(drag.mach_max + 0.02, h) / drag.aero.kts

    params = drag.clean_drag_polar_params(tas=tas, alt=alt)

    v = tas * drag.aero.kts
    rho = drag.aero.density(h)
    qS = 0.5 * rho * v**2 * drag.S
    cl = mass * drag.aero.g0 / max(qS, 1e-3)
    expected = qS * (params["cd0"] + params["cd2"] * cl**2 + params["cd6"] * cl**6)

    assert drag.clean(mass=mass, tas=tas, alt=alt) == pytest.approx(expected)


def test_bada4_idle_thrust_polynomial_matches_original_einsum_formula(bada4_path):
    thrust = bada4.Thrust("A320-TEST", bada4_path)
    mach = np.array([[0.45]])
    h = np.array([[12000.0 * thrust.aero.ft]])

    delta = thrust.aero.pressure(h) / thrust.aero.p0
    ti_matrix = np.reshape(thrust.ti, (3, 4))
    delta_pow = np.array([delta**i for i in range(-1, 3)]).reshape(4, -1)
    mach_pow = np.array([mach**i for i in range(3)]).reshape(3, -1)
    expected = np.einsum("ij,jk,ik->k", ti_matrix, delta_pow, mach_pow)

    result = thrust.cT(mach, h, "LIDL", dT=0.0)

    assert result == pytest.approx(float(expected[0]))


def test_bada4_temp_rating_polynomial_matches_original_einsum_formula(bada4_path):
    thrust = bada4.Thrust("A320-TEST", bada4_path)
    mach = np.array([[0.45]])
    h = np.array([[12000.0 * thrust.aero.ft]])
    dT = 20.0
    k = 1.4

    delta = thrust.aero.pressure(h, dT=dT) / thrust.aero.p0
    theta = thrust.aero.temperature(h, dT=dT) / thrust.aero.T0
    c_matrix = np.reshape(thrust.c_["MCMB"], (9, 5))
    mach_pow = np.array([mach**i for i in range(5)]).reshape(5, -1)
    theta_t = theta * (1 + (mach**2) * (k - 1) / 2)
    ratio_pow = np.array(
        [theta_t**j for j in range(5)] + [delta**j for j in range(1, 5)]
    ).reshape(9, -1)
    delta_t = np.einsum("ij,jk,ik->k", c_matrix, mach_pow, ratio_pow)

    a_matrix = np.reshape(thrust.a_, (6, 6))
    mach_pow = np.array([mach**i for i in range(6)]).reshape(6, -1)
    delta_t_pow = np.array([delta_t**j for j in range(6)]).reshape(6, -1)
    expected = np.einsum("ij,jk,ik->k", a_matrix, mach_pow, delta_t_pow)

    result = thrust.cT(mach, h, "MCMB", dT=dT)

    assert result == pytest.approx(float(expected[0]))


def test_bada4_fuel_polynomials_match_bada_turbofan_formulas(bada4_path):
    fuel_flow = bada4.FuelFlow("A320-TEST", bada4_path)
    mass = np.array([[60000.0]])
    tas = np.array([[300.0]])
    alt = np.array([[12000.0]])
    vs = np.array([[500.0]])

    h = alt * fuel_flow.aero.ft
    v = tas * fuel_flow.aero.kts
    mach = fuel_flow.aero.tas2mach(v, h)
    delta = fuel_flow.aero.pressure(h) / fuel_flow.aero.p0
    theta = fuel_flow.aero.temperature(h) / fuel_flow.aero.T0

    fi_matrix = np.reshape(fuel_flow.fi_, (3, 3))
    delta_powers = np.array([delta**i for i in range(3)]).reshape(3, -1)
    mach_powers = np.array([mach**i for i in range(3)]).reshape(3, -1)
    cF_idle = (
        np.einsum("ij,jk,ik->k", fi_matrix, delta_powers, mach_powers)
        * delta.reshape(-1) ** -1
        * theta.reshape(-1) ** -0.5
    )
    expected_idle = fuel_flow._calc_fuel(mass, delta, theta, cF_idle)

    gamma = np.arctan2(vs * fuel_flow.aero.fpm, v)
    drag = fuel_flow.drag.clean(mass, tas, alt, vs)
    thrust = drag + mass * fuel_flow.aero.g0 * np.sin(gamma)
    cT = thrust / (delta.reshape(-1, 1) * fuel_flow.mass_ref * fuel_flow.aero.g0)

    f_matrix = np.reshape(fuel_flow.f_, (5, 5))
    cT_powers = np.array([cT[:, 0] ** i for i in range(5)]).reshape(5, -1)
    mach_powers = np.array([mach[:, 0] ** i for i in range(5)]).reshape(5, -1)
    cF_gen = np.einsum("ij,jk,ik->k", f_matrix, cT_powers, mach_powers)
    expected_enroute = fuel_flow._calc_fuel(mass, delta, theta, cF_gen)

    assert fuel_flow.idle(mass, tas, alt) == pytest.approx(expected_idle)
    assert fuel_flow.enroute(mass, tas, alt, vs) == pytest.approx(expected_enroute)


def test_bada4_fuel_enroute_supports_casadi_symbolics(casadi, bada4_path):
    numeric = bada4.FuelFlow("A320-TEST", bada4_path).enroute(
        mass=60000.0, tas=300.0, alt=12000.0, vs=500.0
    )

    fuel_flow = bada4.FuelFlow("A320-TEST", bada4_path, backend=CasadiBackend())
    mass = casadi.SX.sym("mass")
    tas = casadi.SX.sym("tas")
    alt = casadi.SX.sym("alt")
    vs = casadi.SX.sym("vs")

    result = _eval1(
        casadi,
        fuel_flow.enroute(mass, tas, alt, vs),
        (mass, 60000),
        (tas, 300),
        (alt, 12000),
        (vs, 500),
    )

    assert result == pytest.approx(numeric)


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("vertical_rate", [-251.0, -300.0, -1500.0])
def test_bada4_powered_descent_uses_required_thrust_fuel(
    casadi, bada4_path, smooth, symbolic, vertical_rate
):
    options = {"backend": CasadiBackend()} if symbolic else {}
    fuel_flow = bada4.FuelFlow("A320-TEST", bada4_path, smooth=smooth, **options)
    mass, tas, alt = 60000.0, 300.0, 12000.0
    idle = float(fuel_flow.idle(mass, tas, alt))
    level = float(fuel_flow.enroute(mass, tas, alt, vs=0.0))
    vs = casadi.MX.sym("vs") if symbolic else vertical_rate
    result = fuel_flow.enroute(mass, tas, alt, vs)
    if symbolic:
        result = float(casadi.Function("fuel", [vs], [result])(vertical_rate))
    assert idle < float(result) < level


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("vertical_rate", [-1500.0, 0.0, 500.0])
def test_bada4_enroute_respects_idle_fuel_floor(
    casadi, bada4_path, smooth, symbolic, vertical_rate
):
    options = {"backend": CasadiBackend()} if symbolic else {}
    fuel_flow = bada4.FuelFlow("A320-TEST", bada4_path, smooth=smooth, **options)
    fuel_flow.f_ = [0.0] * 25
    vs = casadi.MX.sym("vs") if symbolic else vertical_rate
    result = fuel_flow.enroute(60000.0, 300.0, 12000.0, vs)
    if symbolic:
        result = float(casadi.Function("fuel", [vs], [result])(vertical_rate))
    idle = float(fuel_flow.idle(60000.0, 300.0, 12000.0))
    assert idle <= float(result) <= idle + 5e-4


def test_bada4_smooth_fuel_supports_symbolic_derivative(casadi, bada4_path):
    fuel_flow = bada4.FuelFlow(
        "A320-TEST", bada4_path, backend=CasadiBackend(), smooth=True
    )
    vs = casadi.SX.sym("vs")
    ff = fuel_flow.enroute(60000.0, 300.0, 12000.0, vs)
    derivative = casadi.jacobian(ff, vs)
    function = casadi.Function("fuel", [vs], [ff, derivative])
    for vertical_rate in (-251.0, -250.0, -249.0):
        value, slope = function(vertical_rate)
        assert float(value) > 0
        assert np.isfinite(float(slope))


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("symbolic", [False, True])
def test_bada3_nominal_respects_idle_floor_in_kg_s(
    casadi, bada3_model, smooth, symbolic
):
    """Descent fuel must not fall below the BADA idle rate."""
    bada3_model["CfDes"] = [10.0, 80000.0]
    options = {"backend": CasadiBackend()} if symbolic else {}
    fuel_flow = bada3.FuelFlow(
        "A320", bada_path="", model=bada3_model, smooth=smooth, **options
    )
    vs = casadi.MX.sym("vs") if symbolic else -4000.0
    nominal = fuel_flow.nominal(60000.0, 300.0, 12000.0, vs)
    if symbolic:
        nominal = float(casadi.Function("nominal", [vs], [nominal])(-4000.0))
    idle_kg_s = 10.0 * (1.0 - 12000.0 / 80000.0) / 60.0
    assert float(nominal) >= idle_kg_s - 1e-12
    if smooth:
        assert float(nominal) <= idle_kg_s + 0.0025
    else:
        assert float(nominal) == pytest.approx(idle_kg_s)


@pytest.mark.parametrize("kind", ["bada3", "bada4"])
def test_bada_fraction_mass_and_mtow_validation(kind, bada3_model, bada4_path):
    if kind == "bada3":
        drag = bada3.Drag("A320", bada_path=None, model=bada3_model)
        fuel = bada3.FuelFlow("A320", model=bada3_model)
    else:
        drag = bada4.Drag("A320-TEST", bada_path=str(bada4_path))
        fuel = bada4.FuelFlow("A320-TEST", bada_path=str(bada4_path))
    for method in (drag.clean, fuel.enroute, fuel.idle):
        assert float(method(0.87, 420, 30000, vs=500)) == pytest.approx(
            float(method(0.87 * fuel.mtow, 420, 30000, vs=500))
        )
        with pytest.raises(ValueError, match="MTOW"):
            method(fuel.mtow + 1, 420, 30000)
