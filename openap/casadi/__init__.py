"""CasADi interface for OpenAP.

This module provides CasADi-enabled versions of OpenAP models for use
in trajectory optimization (e.g., with openap-top). All models use the
CasadiBackend for symbolic computation and automatic differentiation.

Usage:
    from openap.casadi import Thrust, Drag, FuelFlow, Emission

    # These classes automatically use CasadiBackend
    thrust = Thrust('A320')
    drag = Drag('A320')
    fuelflow = FuelFlow('A320')
    emission = Emission('A320')
"""

from openap import prop
from openap.aero import Aero
from openap.backends import CasadiBackend

# Create a shared CasADi backend instance
_casadi_backend = CasadiBackend()

# Import the base classes
from openap.drag import Drag as _DragBase
from openap.emission import Emission as _EmissionBase
from openap.fuel import FuelFlow as _FuelFlowBase
from openap.thrust import Thrust as _ThrustBase


class Drag(_DragBase):
    """CasADi-enabled drag model."""

    def __init__(self, ac, **kwargs):
        super().__init__(ac, backend=_casadi_backend, **kwargs)


class Thrust(_ThrustBase):
    """CasADi-enabled thrust model."""

    def __init__(self, ac, eng=None, **kwargs):
        super().__init__(ac, eng, backend=_casadi_backend, **kwargs)


class FuelFlow(_FuelFlowBase):
    """CasADi-enabled fuel flow model."""

    def __init__(self, ac, eng=None, **kwargs):
        super().__init__(ac, eng, backend=_casadi_backend, **kwargs)


class Emission(_EmissionBase):
    """CasADi-enabled emission model."""

    def __init__(self, ac, eng=None, **kwargs):
        super().__init__(ac, eng, backend=_casadi_backend, **kwargs)


# Export the CasADi-specific aero module for backward compatibility
aero = Aero(backend=_casadi_backend)

__all__ = ["Drag", "Thrust", "FuelFlow", "Emission", "aero", "prop"]
