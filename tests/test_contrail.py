"""Comprehensive tests for the contrail module.

This module tests all contrail-related functions in openap.contrail,
including saturation pressure, relative humidity, critical temperature,
radiative forcing, and optical property evolution.
"""

import numpy as np
import pytest

from openap import contrail


# Tolerance for floating point comparisons
RTOL = 1e-4  # 0.01% relative tolerance


class TestContrailSaturationPressure:
    """Tests for saturation pressure functions."""

    def test_saturation_pressure_over_water_freezing(self):
        """Test saturation pressure over water at freezing point."""
        # At 273.15 K, saturation pressure should be about 611 Pa
        p_sat = contrail.saturation_pressure_over_water(273.15)
        assert p_sat == pytest.approx(611.0, rel=0.02)

    def test_saturation_pressure_over_water_array(self):
        """Test saturation pressure over water with array input."""
        temperatures = np.array([253.15, 263.15, 273.15])
        p_sat = contrail.saturation_pressure_over_water(temperatures)

        assert isinstance(p_sat, np.ndarray)
        assert p_sat.shape == (3,)
        # Pressure should increase with temperature
        assert p_sat[0] < p_sat[1] < p_sat[2]

    def test_saturation_pressure_over_ice_freezing(self):
        """Test saturation pressure over ice at freezing point."""
        # At 273.15 K, saturation pressure over ice should be about 611 Pa
        p_sat = contrail.saturation_pressure_over_ice(273.15)
        assert p_sat == pytest.approx(611.0, rel=0.02)

    def test_saturation_pressure_over_ice_array(self):
        """Test saturation pressure over ice with array input."""
        temperatures = np.array([223.15, 243.15, 263.15])
        p_sat = contrail.saturation_pressure_over_ice(temperatures)

        assert isinstance(p_sat, np.ndarray)
        assert p_sat.shape == (3,)
        # Pressure should increase with temperature
        assert p_sat[0] < p_sat[1] < p_sat[2]

    def test_water_greater_than_ice(self):
        """Test that saturation pressure over water > ice at same temp."""
        T = 253.15  # -20 C
        p_water = contrail.saturation_pressure_over_water(T)
        p_ice = contrail.saturation_pressure_over_ice(T)

        # Saturation pressure over water is always higher than over ice
        # for the same temperature (supercooled water)
        assert p_water > p_ice


class TestContrailRelativeHumidity:
    """Tests for relative humidity functions."""

    def test_relative_humidity_ice(self):
        """Test relative humidity calculation with respect to ice."""
        # Typical upper troposphere conditions
        specific_humidity = 0.0001  # kg/kg
        pressure = 25000  # Pa (~10 km altitude)
        temperature = 220  # K

        rhi = contrail.relative_humidity(
            specific_humidity, pressure, temperature, to="ice"
        )

        # Should be a reasonable value
        assert 0 < rhi < 3  # Can be supersaturated

    def test_relative_humidity_water(self):
        """Test relative humidity calculation with respect to water."""
        specific_humidity = 0.0001
        pressure = 25000
        temperature = 220

        rhw = contrail.relative_humidity(
            specific_humidity, pressure, temperature, to="water"
        )

        # Should be a reasonable value
        assert 0 < rhw < 2

    def test_relative_humidity_array(self):
        """Test relative humidity with array inputs."""
        specific_humidity = np.array([0.0001, 0.0002, 0.0003])
        pressure = 25000
        temperature = 220

        rhi = contrail.relative_humidity(
            specific_humidity, pressure, temperature, to="ice"
        )

        assert isinstance(rhi, np.ndarray)
        assert rhi.shape == (3,)
        # Higher specific humidity should give higher RH
        assert rhi[0] < rhi[1] < rhi[2]

    def test_relative_humidity_invalid_reference(self):
        """Test that invalid reference phase raises error."""
        with pytest.raises(AssertionError):
            contrail.relative_humidity(0.0001, 25000, 220, to="steam")

    def test_rhw2rhi(self):
        """Test conversion from RH water to RH ice."""
        T = 220  # K
        rhw = 0.5

        rhi = contrail.rhw2rhi(rhw, T)

        # At cold temperatures, RHi should be higher than RHw
        assert rhi > rhw

    def test_rhw2rhi_array(self):
        """Test rhw2rhi with array inputs."""
        temperatures = np.array([210, 220, 230])
        rhw = 0.5

        rhi = contrail.rhw2rhi(rhw, temperatures)

        assert isinstance(rhi, np.ndarray)
        assert rhi.shape == (3,)


class TestContrailCriticalTemperature:
    """Tests for critical temperature functions."""

    def test_critical_temperature_water(self):
        """Test critical temperature calculation."""
        # At typical cruise altitude (~10 km, ~25000 Pa)
        pressure = 25000

        t_crit = contrail.critical_temperature_water(pressure)

        # Critical temperature should be around 220-230 K at cruise altitude
        assert 215 < t_crit < 235

    def test_critical_temperature_water_altitude_dependence(self):
        """Test that critical temperature varies with pressure."""
        p_low = 20000  # Higher altitude
        p_high = 30000  # Lower altitude

        t_crit_low = contrail.critical_temperature_water(p_low)
        t_crit_high = contrail.critical_temperature_water(p_high)

        # Higher altitude (lower pressure) -> lower critical temperature
        assert t_crit_low < t_crit_high

    def test_critical_temperature_water_efficiency_dependence(self):
        """Test that critical temperature varies with propulsion efficiency."""
        pressure = 25000

        t_crit_low_eff = contrail.critical_temperature_water(
            pressure, propulsion_efficiency=0.3
        )
        t_crit_high_eff = contrail.critical_temperature_water(
            pressure, propulsion_efficiency=0.5
        )

        # Higher efficiency -> higher critical temperature
        assert t_crit_high_eff > t_crit_low_eff

    def test_critical_temperature_water_array(self):
        """Test critical temperature with array input."""
        pressures = np.array([20000, 25000, 30000])

        t_crit = contrail.critical_temperature_water(pressures)

        assert isinstance(t_crit, np.ndarray)
        assert t_crit.shape == (3,)

    def test_critical_temperature_water_and_ice(self):
        """Test critical temperature for both water and ice."""
        pressure = 25000

        t_water, t_ice = contrail.critical_temperature_water_and_ice(pressure)

        # Ice critical temperature should be lower than water
        assert t_ice < t_water

    def test_critical_temperature_water_and_ice_efficiency(self):
        """Test critical temperatures with custom efficiency."""
        pressure = 25000

        t_water, t_ice = contrail.critical_temperature_water_and_ice(
            pressure, propulsion_efficiency=0.35
        )

        assert t_ice < t_water
        assert 200 < t_ice < 230
        assert 210 < t_water < 240

    def test_backward_compatibility_propulsion_efficiency(self):
        """Test that module-level propulsion_efficiency constant exists."""
        # This ensures backward compatibility
        assert contrail.propulsion_efficiency == 0.4
        assert contrail.DEFAULT_PROPULSION_EFFICIENCY == 0.4


class TestContrailRadiativeForcing:
    """Tests for radiative forcing functions."""

    def test_rf_shortwave_daytime(self):
        """Test shortwave radiative forcing during daytime."""
        zenith = 30  # degrees
        tau = 0.4
        tau_c = 0.36

        rf_sw = contrail.rf_shortwave(zenith, tau, tau_c)

        # Shortwave forcing should be negative (cooling)
        assert rf_sw < 0

    def test_rf_shortwave_nighttime(self):
        """Test shortwave radiative forcing at night."""
        zenith = 100  # degrees (sun below horizon)
        tau = 0.4
        tau_c = 0.36

        rf_sw = contrail.rf_shortwave(zenith, tau, tau_c)

        # No shortwave forcing at night
        assert rf_sw == 0

    def test_rf_shortwave_horizon(self):
        """Test shortwave forcing at horizon."""
        zenith = 90  # exactly at horizon
        tau = 0.4
        tau_c = 0.36

        rf_sw = contrail.rf_shortwave(zenith, tau, tau_c)

        # Very small or zero forcing at horizon
        assert abs(rf_sw) < 1.0

    def test_rf_shortwave_array(self):
        """Test shortwave forcing with array inputs."""
        zenith = np.array([30, 60, 100])
        tau = 0.4
        tau_c = 0.36

        rf_sw = contrail.rf_shortwave(zenith, tau, tau_c)

        assert isinstance(rf_sw, np.ndarray)
        assert rf_sw.shape == (3,)
        assert rf_sw[0] < 0  # Daytime cooling
        assert rf_sw[1] < 0  # Daytime cooling
        assert rf_sw[2] == 0  # Nighttime

    def test_rf_longwave(self):
        """Test longwave radiative forcing."""
        olr = 250  # W/m^2
        temperature = 220  # K

        rf_lw = contrail.rf_longwave(olr, temperature)

        # Longwave forcing should be positive (warming)
        assert rf_lw > 0

    def test_rf_longwave_array(self):
        """Test longwave forcing with array inputs."""
        olr = np.array([200, 250, 300])
        temperature = 220

        rf_lw = contrail.rf_longwave(olr, temperature)

        assert isinstance(rf_lw, np.ndarray)
        assert rf_lw.shape == (3,)
        # Higher OLR should give higher forcing
        assert rf_lw[0] < rf_lw[1] < rf_lw[2]

    def test_rf_longwave_nonnegative(self):
        """Test that longwave forcing is non-negative."""
        # Edge case: very low OLR
        rf_lw = contrail.rf_longwave(0, 220)
        assert rf_lw >= 0

    def test_rf_net(self):
        """Test net radiative forcing calculation."""
        zenith = 30
        tau = 0.4
        tau_c = 0.36
        olr = 250
        temperature = 220

        rf_total = contrail.rf_net(zenith, tau, tau_c, olr, temperature)
        rf_sw = contrail.rf_shortwave(zenith, tau, tau_c)
        rf_lw = contrail.rf_longwave(olr, temperature)

        # Net should equal SW + LW
        assert rf_total == pytest.approx(rf_sw + rf_lw, rel=RTOL)

    def test_rf_net_nighttime_warming(self):
        """Test that net forcing is warming at night."""
        zenith = 100  # Night
        tau = 0.4
        tau_c = 0.36
        olr = 250
        temperature = 220

        rf_total = contrail.rf_net(zenith, tau, tau_c, olr, temperature)

        # At night, only LW forcing, which is positive
        assert rf_total > 0


class TestContrailOpticalEvolution:
    """Tests for contrail optical property evolution."""

    def test_contrail_optical_properties_young(self):
        """Test optical properties for young contrail (0-1 hours)."""
        tau, width, tau_c = contrail.contrail_optical_properties(0.5)

        assert tau == 0.4
        assert width == 500
        assert tau_c == 0.36

    def test_contrail_optical_properties_aged(self):
        """Test optical properties for aged contrail (6+ hours)."""
        tau, width, tau_c = contrail.contrail_optical_properties(8.0)

        assert tau == 0.71
        assert width == 10500
        assert tau_c == 0.639

    def test_contrail_optical_properties_intermediate(self):
        """Test optical properties at intermediate ages."""
        # 1-2 hours
        tau, width, tau_c = contrail.contrail_optical_properties(1.5)
        assert tau == 0.6
        assert width == 1500
        assert tau_c == 0.54

        # 2-4 hours
        tau, width, tau_c = contrail.contrail_optical_properties(3.0)
        assert tau == 0.68
        assert width == 3500
        assert tau_c == 0.612

        # 4-6 hours
        tau, width, tau_c = contrail.contrail_optical_properties(5.0)
        assert tau == 0.70
        assert width == 6500
        assert tau_c == 0.63

    def test_contrail_optical_properties_array(self):
        """Test optical properties with array input."""
        ages = np.array([0.5, 1.5, 3.0, 5.0, 8.0])

        tau, width, tau_c = contrail.contrail_optical_properties(ages)

        assert isinstance(tau, np.ndarray)
        assert isinstance(width, np.ndarray)
        assert isinstance(tau_c, np.ndarray)
        assert tau.shape == (5,)

        # Values should be monotonically increasing with age
        assert np.all(np.diff(tau) >= 0)
        assert np.all(np.diff(width) >= 0)
        assert np.all(np.diff(tau_c) >= 0)

    def test_contrail_optical_properties_scalar_return(self):
        """Test that scalar input returns scalar outputs."""
        tau, width, tau_c = contrail.contrail_optical_properties(2.5)

        assert isinstance(tau, float)
        assert isinstance(width, float)
        assert isinstance(tau_c, float)


class TestContrailLoadOLR:
    """Tests for OLR data loading."""

    def test_load_olr_import_error(self):
        """Test that load_olr raises ImportError if xarray not available."""
        # This test only works if xarray is not installed
        # If xarray is installed, this test is skipped
        try:
            import xarray

            pytest.skip("xarray is installed, cannot test ImportError")
        except ImportError:
            with pytest.raises(ImportError, match="xarray is required"):
                contrail.load_olr("fake_file.nc", 0, 0, None)


class TestContrailModuleConstants:
    """Tests for module constants."""

    def test_physical_constants(self):
        """Test that physical constants have correct values."""
        assert contrail.gas_constant_water_vapor == 461.51
        assert contrail.gas_constant_dry_air == 287.05
        assert contrail.ei_water == 1.2232
        assert contrail.spec_combustion_heat == 43e6

    def test_default_propulsion_efficiency(self):
        """Test default propulsion efficiency value."""
        assert contrail.DEFAULT_PROPULSION_EFFICIENCY == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
