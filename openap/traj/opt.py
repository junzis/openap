import numpy as np
from openap import aero, prop, Thrust, Drag, FuelFlow
from scipy.optimize import minimize
from numdifftools import Jacobian


def calc_normfactor(x):
    return 1 / x


def denormalize(x, normfactor):
    return x / normfactor


class CruiseOptimizer(object):
    """Optimizer for cursie mach number and altitude."""

    def __init__(self, ac):
        super(CruiseOptimizer, self).__init__()

        self.ac = ac
        self.aircraft = prop.aircraft(ac)
        self.thrust = Thrust(ac)
        self.fuelflow = FuelFlow(ac)
        self.drag = Drag(ac)

        # parameters to be optimized:
        #   Mach number, altitude
        self.x0 = np.array([0.3, 25000 * aero.ft])
        self.normfactor = calc_normfactor(self.x0)

        self.bounds = None
        self.update_bounds()

    def update_bounds(self, **kwargs):
        machmin = kwargs.get("machmin", 0.5)
        machmax = kwargs.get("machmax", self.aircraft["limits"]["MMO"])
        hmin = kwargs.get("hmin", 25000 * aero.ft)
        hmax = kwargs.get("hmax", self.aircraft["limits"]["ceiling"])

        self.bounds = np.array(
            [[machmin, machmax], [hmin, hmax]]
        ) * self.normfactor.reshape(2, -1)

    def func_fuel(self, x, mass):
        mach, h = denormalize(x, self.normfactor)
        va = aero.mach2tas(mach, h)
        ff = self.fuelflow.enroute(mass, va / aero.kts, h / aero.ft)
        ff_m = ff / (va + 1e-3) * 1000
        # print("%.03f" % mach, "%d" % (h/aero.ft), "%.05f" % ff_m)
        return ff_m

    def func_time(self, x, mass):
        mach, h = denormalize(x, self.normfactor)
        va = aero.mach2tas(mach, h)
        va_inv = 1 / (va + 1e-4) * 1000
        # print("%.03f" % mach, "%d" % (h/aero.ft), "%.02f" % va)
        return va_inv

    def func_cons_lift(self, x, mass):
        mach, h = denormalize(x, self.normfactor)
        va = aero.mach2tas(mach, h)

        Tmax = self.thrust.cruise(va / aero.kts, h / aero.ft)

        qS = 0.5 * aero.density(h) * va ** 2 * self.aircraft["wing"]["area"]
        cd0 = self.drag.polar["clean"]["cd0"]
        k = self.drag.polar["clean"]["k"]

        mach_crit = self.drag.polar["mach_crit"]
        if mach > mach_crit:
            cd0 += 20 * (mach - mach_crit) ** 4

        dL2 = qS ** 2 * (1 / k * (Tmax / (qS + 1e-3) - cd0)) - (mass * aero.g0) ** 2
        return dL2

    def func_cons_thrust(self, x, mass):
        mach, h = denormalize(x, self.normfactor)
        va = aero.mach2tas(mach, h)

        D = self.drag.clean(mass, va / aero.kts, h / aero.ft)
        Tmax = self.thrust.cruise(va / aero.kts, h / aero.ft)

        dT = Tmax - D
        return dT

    def optimize(self, goal, mass):
        if goal == "fuel":
            func = self.func_fuel
        elif goal == "time":
            func = self.func_time
        else:
            raise RuntimeError("Optimization goal [%s] not avaiable." % goal)

        x0 = self.x0 * self.normfactor
        res = minimize(
            func,
            x0,
            args=(mass,),
            bounds=self.bounds,
            jac=lambda x, m: Jacobian(lambda x: func(x, m))(x),
            options={"maxiter": 200},
            constraints=(
                {
                    "type": "ineq",
                    "args": (mass,),
                    "fun": lambda x, m: self.func_cons_thrust(x, m),
                    "jac": lambda x, m: Jacobian(lambda x: self.func_cons_thrust(x, m))(
                        x
                    ),
                },
                {
                    "type": "ineq",
                    "args": (mass,),
                    "fun": lambda x, m: self.func_cons_lift(x, m),
                    "jac": lambda x, m: Jacobian(lambda x: self.func_cons_lift(x, m))(
                        x
                    ),
                },
            ),
        )
        return res


if __name__ == "__main__":

    opt = CruiseOptimizer("A320")

    # opt.update_bounds(hmin=30000 * aero.ft, hmax=37000 * aero.ft)

    m = 68000

    res = opt.optimize("fuel", mass=m)
    if res.success:
        print("success", "mass=%d" % m, end=" ")
    else:
        print(res.message)
        print("fail", "mass=%d" % m, end=" ")

    x = denormalize(res.x, opt.normfactor)
    print("mach=%.03f" % x[0], "alt=%d" % (x[1] / aero.ft))

    # print('--------------------- Fuel ---------------------')
    # for m in np.linspace(45000, 78000, 30):
    #     res = opt.optimize('fuel', mass=m)
    #     if res.success:
    #         print('success', 'mass=%d' % m, end=' ')
    #     else:
    #         print(res.message)
    #         print("fail", 'mass=%d' % m, end=' ')

    #     x = denormalize(res.x, opt.normfactor)
    #     print('mach=%.03f' % x[0], 'alt=%d' % (x[1]/aero.ft))

    # print('--------------------- Time ---------------------')
    # for m in np.linspace(45000, 78000, 30):
    #     res = opt.optimize('time', mass=m)
    #     if res.success:
    #         print('success', 'mass=%d' % m, end=' ')
    #     else:
    #         print("finished", 'mass=%d' % m, res.message, end=' ')

    #     x = denormalize(res.x, opt.normfactor)
    #     print('mach=%.03f' % x[0], 'alt=%d' % (x[1]/aero.ft))