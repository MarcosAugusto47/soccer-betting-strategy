import warnings

from time import time
from scipy.optimize import minimize


class TookTooLong(Warning):
    pass


class Optimizer:

    def __init__(self, maxtime_sec=120):
        self.nit = 0
        self.maxtime_sec = maxtime_sec
        self.flag=False

    def callback(self, x):
        # callback to terminate if maxtime_sec is exceeded
        self.nit += 1
        elapsed_time = time() - self.start_time
        
        if elapsed_time > self.maxtime_sec:

            self.flag=True
            
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
            
        else:
            print("Elapsed: %.3f sec" % elapsed_time)
            #print("Elapsed iterations: ", self.nit)

    def run_optimization(self, fun, x0, args):
        self.start_time = time()
        
        res = minimize(fun=fun,
                       x0=x0,
                       args=args,
                       callback=self.callback,
                       tol=0.01,
                       method='Powell',
        )
        
        return res.x, self.flag
