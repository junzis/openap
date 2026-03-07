# OpenAP: Open Aircraft Performance Model and Toolkit

[![PyPI version](https://img.shields.io/pypi/v/openap.svg)](https://pypi.org/project/openap/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/openap.svg)](https://pypi.org/project/openap/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Open-source aircraft performance model and toolkit for Python. Calculate aircraft performance, fuel consumption, and emissions for air transportation studies and simulations.

**Requirements:** Python 3.11+

## 🕮 User Guide

The OpenAP handbook is available at [openap.dev](https://openap.dev/).

## Installation

Install the latest stable release from PyPI:

```sh
pip install --upgrade openap
```

Install the development branch from GitHub (may not be stable):

```sh
pip install --upgrade git+https://github.com/junzis/openap
```

## Content

### Model Data

Data in this repository includes:

- Aircraft data: Collected from open literature.
- Engine data: Primarily from the ICAO emission data-bank, including fuel flow and emissions.
- Drag polar model data: Exclusively derived from open data ([reference](https://research.tudelft.nl/files/71038050/published_OpenAP_drag_polar.pdf)).
- Fuel model data: Polynomial models derived from the [acropole model](https://github.com/DGAC/Acropole) by [@JarryGabriel](https://github.com/JarryGabriel).
- Kinematic data: The kinematic model describes speed, altitude, and vertical rate ([reference](https://github.com/junzis/wrap)).
- Navigation data: Airport and waypoints obtained from [X-Plane](https://developer.x-plane.com/docs/data-development-documentation/).

### Python Packages

The OpenAP Python library includes the following packages:

- `prop`: Module for accessing aircraft and engine properties.
- `aero`: Module for common aeronautical conversions.
- `nav`: Module for accessing navigation information.
- `thrust`: Module provides `Thrust()` class for computing aircraft thrust.
- `drag`: Module provides `Drag()` class for computing aircraft drag.
- `fuel`: Module provides `FuelFlow()` class for computing fuel consumption.
- `emission`: Module provides `Emission()` class for computing aircraft emissions.
- `kinematic`: Module provides `WRAP()` class for accessing kinematic performance data.
- `phase`: Module provides `FlightPhase()` class for determining flight phases.
- `gen`: Module provides `FlightGenerator()` class for trajectory generation.

Example:

```python
import openap

# Get aircraft properties
aircraft = openap.prop.aircraft("A320")
print(aircraft["mtow"])  # max takeoff weight: 78000 (kg)

# Calculate fuel flow during cruise
fuelflow = openap.FuelFlow("A320")
ff = fuelflow.enroute(mass=60000, tas=250, alt=30000)
print(ff)  # fuel flow: 0.92 (kg/s)
```

**Units:** Input parameters can be scalar, list, or ndarray. Speeds are in knots, altitudes in feet, vertical rates in feet/min. Mass is in kilograms (SI).

### Add-ons

The OpenAP library can also be used to interact with BADA performance models if you have access to the BADA data from EUROCONTROL. You can use the following code:

```python
from openap.addon import bada4

fuelflow = bada4.FuelFlow()
```

The methods and attributes of `openap.addon.bada4.FuelFlow()` are the same as those of `openap.FuelFlow()`.

## Alternative Backends: CasADi and JAX

OpenAP supports multiple computational backends beyond NumPy:

- **CasADi**: For symbolic computations and optimization
- **JAX**: For automatic differentiation and GPU acceleration

### Installation

Install with optional backend support:

```sh
pip install openap[casadi]  # CasADi backend
pip install openap[jax]     # JAX backend
pip install openap[all]     # Both backends
```

### Usage

```python
# CasADi backend
import openap.casadi as oc

fuelflow = oc.FuelFlow("A320")
fuelflow.enroute(mass, tas, alt)  # works with CasADi DM, SX, or MX types

# JAX backend
import openap.jax as oj

fuelflow = oj.FuelFlow("A320")
fuelflow.enroute(mass, tas, alt)  # works with JAX arrays
```

The API is identical to the standard `openap` module. The backends are implemented using a protocol-based architecture in `openap/backends/`.

## Citing OpenAP

```
@article{sun2020openap,
  title={OpenAP: An open-source aircraft performance model for air transportation studies and simulations},
  author={Sun, Junzi and Hoekstra, Jacco M and Ellerbroek, Joost},
  journal={Aerospace},
  volume={7},
  number={8},
  pages={104},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
