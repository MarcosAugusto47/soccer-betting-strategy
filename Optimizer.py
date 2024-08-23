import warnings
from time import time

from scipy.optimize import minimize


class TookTooLong(Warning):
    pass


class Optimizer:
    def __init__(self, maxtime_sec=120):
        self.nit = 0
        self.maxtime_sec = maxtime_sec
        self.flag = False

    def callback(self, x):
        # callback to terminate if maxtime_sec is exceeded
        self.nit += 1
        elapsed_time = time() - self.start_time

        if elapsed_time > self.maxtime_sec:
            self.flag = True

            warnings.warn("Terminating optimization: time limit reached", TookTooLong)

        # else:
        # print("Elapsed: %.3f sec" % elapsed_time)
        # print("Elapsed iterations: ", self.nit)

    def run_optimization(self, fun, x0, args, bounds=None):
        self.start_time = time()

        # # Constraint function
        # def constraint_equation(x):
        #     # This function returns 0 when the sum of x equals 1
        #     return sum(x) - 1

        # # Constraint dictionary
        # constraint = {'type': 'eq', 'fun': constraint_equation}

        res = minimize(
            fun=fun,
            x0=x0,
            args=args,
            bounds=bounds,
            # callback=self.callback,
            tol=0.001,
            method="COBYLA",
            # constraints=constraint,
            options={"disp": True},
        )

        return res.x, res.fun, self.flag
