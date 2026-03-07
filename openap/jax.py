"""JAX interface for OpenAP.

This module provides JAX-enabled versions of OpenAP models for use
in trajectory optimization, automatic differentiation, and JIT compilation.
All models use the JaxBackend for numerical operations.

Usage:
    from openap.jax import Thrust, Drag, FuelFlow, Emission

    # These classes automatically use JaxBackend
    thrust = Thrust('A320')
    drag = Drag('A320')
    fuelflow = FuelFlow('A320')
    emission = Emission('A320')

    # Can be used with JAX transformations
    import jax

    @jax.jit
    def compute_drag(mass, tas, alt):
        return drag.clean(mass, tas, alt)

    # Compute gradients
    grad_fn = jax.grad(compute_drag)
"""

from openap.aero import Aero
from openap.backends import JaxBackend

# Create a shared JAX backend instance
_jax_backend = JaxBackend()

# Import the base classes
from openap.drag import Drag as _DragBase
from openap.emission import Emission as _EmissionBase
from openap.fuel import FuelFlow as _FuelFlowBase
from openap.thrust import Thrust as _ThrustBase


class Drag(_DragBase):
    """JAX-enabled drag model."""

    def __init__(self, ac, **kwargs):
        super().__init__(ac, backend=_jax_backend, **kwargs)


class Thrust(_ThrustBase):
    """JAX-enabled thrust model."""

    def __init__(self, ac, eng=None, **kwargs):
        super().__init__(ac, eng, backend=_jax_backend, **kwargs)


class FuelFlow(_FuelFlowBase):
    """JAX-enabled fuel flow model."""

    def __init__(self, ac, eng=None, **kwargs):
        super().__init__(ac, eng, backend=_jax_backend, **kwargs)


class Emission(_EmissionBase):
    """JAX-enabled emission model."""

    def __init__(self, ac, eng=None, **kwargs):
        super().__init__(ac, eng, backend=_jax_backend, **kwargs)


# Export the JAX-specific aero module
aero = Aero(backend=_jax_backend)

__all__ = ["Drag", "Emission", "FuelFlow", "Thrust", "aero"]
