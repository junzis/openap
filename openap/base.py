import importlib

from openap.extra import ndarrayconvert


class DragBase(object):
    """Base class for drag models."""

    def __init__(self, ac, **kwargs):
        """Initialize BaseDrag object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).

        """
        if not hasattr(self, "sci"):
            self.sci = importlib.import_module("numpy")

        if not hasattr(self, "aero"):
            self.aero = importlib.import_module("openap").aero

        self.ac = ac.upper()

    def clean(self, mass, tas, alt, vs):
        raise NotImplementedError

    def nonclean(self, mass, tas, alt, flap_angle, vs=0, landing_gear=False):
        raise NotImplementedError


class ThrustBase(object):
    """Base class for thrust models."""

    def __init__(self, ac, eng=None, **kwargs):
        """Initialize ThrustBase object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            eng (string): Engine type (for example: CFM56-5A3).

        """
        if not hasattr(self, "sci"):
            self.sci = importlib.import_module("numpy")

        if not hasattr(self, "aero"):
            self.aero = importlib.import_module("openap").aero

        self.ac = ac.upper()

    def takeoff(self, tas, alt):
        raise NotImplementedError

    def climb(self, tas, alt):
        raise NotImplementedError

    def cruise(self, tas, alt, roc):
        raise NotImplementedError

    def idle(self, tas, alt, roc):
        raise NotImplementedError


class FuelFlowBase(object):
    """Base class for fuel flow models."""

    def __init__(self, ac, eng=None, **kwargs):
        """Initialize FuelFlowBase object.

        Args:
            ac (string): ICAO aircraft type (for example: A320).
            eng (string): Engine type (for example: CFM56-5A3).
                Leave empty to use the default engine specified
                by in the aircraft database.

        """
        if not hasattr(self, "sci"):
            self.sci = importlib.import_module("numpy")

        if not hasattr(self, "aero"):
            self.aero = importlib.import_module("openap").aero

        self.ac = ac.upper()

    @ndarrayconvert
    def enroute(self, mass, tas, alt, vs=0, acc=0):
        raise NotImplementedError

    @ndarrayconvert
    def idle(self, mass, tas, alt, vs=0):
        raise NotImplementedError
