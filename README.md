# Open Aircraft Performance Model (OpenAP) and Toolkit

This repository contains the OpenAP model data and Python packages for aircraft performance and emission calculations.

More information on user guide and related articles at: [OpenAP.dev](https://openap.dev/)

The most recent development fork of OpenAP is: https://github.com/junzis/openap. For discussions and issues, you can use the following links:

 - https://github.com/junzis/openap/discussions (discussions, feedback and suggestions)
 - https://github.com/junzis/openap/issues (bug reporting)


## Install


Install the development branch from GitHub:

```sh
pip install --upgrade git+https://github.com/junzis/openap
```

Install the latest stable release from pypi:

```sh
pip install --upgrade openap
```

Install the latest stable release on conda-forge:

```sh
conda install openap
```

## Content

OpenAP library has two parts, which are __OpenAP model data__ and __OpenAP Python packages__.


### Model data

Data in this repository includes:

  - Aircraft data: Collected from open literature.
  - Engines data: Primarily from ICAO emission data-bank, including fuel flow and emissions.
  - Drag polar data: Exclusively derived from open data ([reference](https://junzis.com/files/openap_dragpolar.pdf)).
  - Kinematic data: Kinematic model (formally [WRAP](https://github.com/junzis/wrap)) describe speed, altitude, and vertical rate.
  - Navigation data: Airport and waypoints obtained from [X-plane ](https://developer.x-plane.com/docs/data-development-documentation/).


### Python packages

The OpenAP Python library includes the following packages:

  - `prop`: a package for accessing aircraft and engine properties
  - `thrust`: a package for computing aircraft thrust
  - `drag`: a package for computing aircraft drag
  - `fuel`: a package for computing fuel consumption
  - `emission`: a package for computing aircraft emissions
  - `kinematic`: a package for accessing WRAP data
  - `aero`: a package for common aeronautical conversions
  - `nav`: a package for accessing navigation information
  - `segment`: a package for determining climb, cruise, descent, level flight
  - `phase`: a wrapper around `segment`, providing identification of all flight phases
  - `traj`: package contains a set of tools for trajectory generation



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
