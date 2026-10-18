"""Base classes for OpenAP performance models.

These base classes provide the foundation for thrust, drag, and fuel flow
models with configurable backends (NumPy, CasADi, JAX).
"""

from functools import cached_property
from typing import Optional

from openap.aero import Aero
from openap.backends import BackendType, default_backend
from openap.extra import mass_to_kg, ndarrayconvert


class _MassModel:
    @cached_property
    def mtow(self):
        from openap import prop

        aircraft = getattr(self, "aircraft", None)
        if aircraft is None:
            aircraft = prop.aircraft(self.ac.split("-", 1)[0])
        return aircraft["mtow"]

    def mass_to_kg(self, mass):
        """Return kg from kg or an MTOW fraction in (0, 1]."""
        return mass_to_kg(mass, self.mtow, backend=self.backend)


class DragBase(_MassModel):
    """Base class for drag models."""

    def __init__(self, ac: str, backend: Optional[BackendType] = None, **kwargs):
        """Initialize DragBase object.

        Args:
            ac: ICAO aircraft type (for example: A320).
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        self.backend = backend or default_backend()
        self.aero = Aero(backend=self.backend)
        self.ac = ac.lower()

    def clean(self, mass, tas, alt, vs):
        raise NotImplementedError

    def nonclean(self, mass, tas, alt, flap_angle, vs=0, landing_gear=False):
        raise NotImplementedError


class ThrustBase:
    """Base class for thrust models."""

    def __init__(
        self,
        ac: str,
        eng: Optional[str] = None,
        backend: Optional[BackendType] = None,
        **kwargs,
    ):
        """Initialize ThrustBase object.

        Args:
            ac: ICAO aircraft type (for example: A320).
            eng: Engine type (for example: CFM56-5A3).
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        self.backend = backend or default_backend()
        self.aero = Aero(backend=self.backend)
        self.ac = ac.lower()

    def takeoff(self, tas, alt):
        raise NotImplementedError

    def climb(self, tas, alt):
        raise NotImplementedError

    def cruise(self, tas, alt, roc):
        raise NotImplementedError

    def idle(self, tas, alt, roc):
        raise NotImplementedError


class FuelFlowBase(_MassModel):
    """Base class for fuel flow models."""

    def __init__(
        self,
        ac: str,
        eng: Optional[str] = None,
        backend: Optional[BackendType] = None,
        **kwargs,
    ):
        """Initialize FuelFlowBase object.

        Args:
            ac: ICAO aircraft type (for example: A320).
            eng: Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                in the aircraft database.
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        self.backend = backend or default_backend()
        self.aero = Aero(backend=self.backend)
        self.ac = ac.lower()

    @ndarrayconvert
    def enroute(self, mass, tas, alt, vs=0, acc=0):
        raise NotImplementedError

    @ndarrayconvert
    def idle(self, mass, tas, alt, vs=0):
        raise NotImplementedError


class EmissionBase:
    """Base class for emission models."""

    def __init__(
        self,
        ac: str,
        eng: Optional[str] = None,
        backend: Optional[BackendType] = None,
        **kwargs,
    ):
        """Initialize EmissionBase object.

        Args:
            ac: ICAO aircraft type (for example: A320).
            eng: Engine type (for example: CFM56-5A3).
            backend: Math backend to use. Defaults to NumpyBackend.
        """
        self.backend = backend or default_backend()
        self.aero = Aero(backend=self.backend)
        self.ac = ac.lower()
