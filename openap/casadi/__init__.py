from .. import *
from . import aero_override as aero
from . import numpy_override as sci


class RemoveDecoratorMeta(type):
    def __new__(cls, name, base, attr_dict):
        # for all methods in all base classes
        # reimplement in attr_dict
        for b in base:
            for elt in vars(b):
                if hasattr(getattr(b, elt), "orig_func"):
                    attr_dict[elt] = getattr(b, elt).orig_func

        attr_dict["sci"] = sci
        attr_dict["aero"] = aero
        return super().__new__(cls, name, base, attr_dict)


class Drag(drag.Drag, metaclass=RemoveDecoratorMeta):
    pass


class Thrust(thrust.Thrust, metaclass=RemoveDecoratorMeta):
    pass


class FuelFlow(fuel.FuelFlow, metaclass=RemoveDecoratorMeta):
    def __init__(self, ac, eng=None, **kwargs):
        self.Drag = Drag
        self.Thrust = Thrust
        super(FuelFlow, self).__init__(ac=ac, eng=eng, **kwargs)


class Emission(emission.Emission, metaclass=RemoveDecoratorMeta):
    pass
