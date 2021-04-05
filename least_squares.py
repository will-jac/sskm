"""
Sample code automatically generated on 2021-03-28 16:24:34

by geno from www.geno-project.org

from input

parameters
  matrix A
  vector b
variables
  vector x
min
  norm2(A*x-b).^2


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
import numpy as np


try:
    from genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)

class GenoNLP:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        assert isinstance(A, np.ndarray)
        dim = A.shape
        assert len(dim) == 2
        self.A_rows = dim[0]
        self.A_cols = dim[1]
        assert isinstance(b, np.ndarray)
        dim = b.shape
        assert len(dim) == 1
        self.b_rows = dim[0]
        self.b_cols = 1
        self.x_rows = self.A_cols
        self.x_cols = 1
        self.x_size = self.x_rows * self.x_cols
        # the following dim assertions need to hold for this problem
        assert self.b_rows == self.A_rows
        assert self.x_rows == self.A_cols

    def getBounds(self):
        bounds = []
        bounds += [(-inf, inf)] * self.x_size
        return bounds

    def getStartingPoint(self):
        self.xInit = np.random.randn(self.x_rows, self.x_cols)
        return self.xInit.reshape(-1)

    def variables(self, _x):
        x = _x
        return x

    def fAndG(self, _x):
        x = self.variables(_x)
        t_0 = ((self.A).dot(x) - self.b)
        f_ = (np.linalg.norm(t_0) ** 2)
        g_0 = (2 * (self.A.T).dot(t_0))
        g_ = g_0
        return f_, g_

def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

def solve(A, b):
    start = timer()
    NLP = GenoNLP(A, b)
    x0 = NLP.getStartingPoint()
    bnds = NLP.getBounds()
    tol = 1E-6
    # These are the standard GENO solver options, they can be omitted.
    options = {'tol' : tol,
               'constraintsTol' : 1E-4,
               'maxiter' : 1000,
               'verbosity' : 1  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.0.3')
        result = minimize(NLP.fAndG, x0,
                          bounds=bnds, options=options)
    else:
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=bnds)

    # assemble solution and map back to original problem
    x = result.x
    x = NLP.variables(x)
    solution = {}
    solution['success'] = result.success
    solution['message'] = result.message
    solution['fun'] = result.fun
    solution['grad'] = result.jac
    solution['x'] = x
    solution['elapsed'] = timer() - start
    return solution

def generateRandomData():
    np.random.seed(0)
    A = np.random.randn(3, 3)
    b = np.random.randn(3)
    return A, b

if __name__ == '__main__':
    print('\ngenerating random instance')
    A, b = generateRandomData()
    print('solving ...')
    solution = solve(A, b)
    print('*'*5, 'solution', '*'*5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable x = ', solution['x'])
        print('solving took %.3f sec' % solution['elapsed'])
