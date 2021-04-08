"""
Sample code automatically generated on 2021-04-08 00:30:56

by geno from www.geno-project.org

from input

parameters
  matrix K symmetric
  matrix A
  matrix B
  matrix M symmetric
  matrix C
  vector y
  scalar g
  scalar ga
  scalar gi
  scalar u
variables
  vector a
min
  norm2(K*a-y).^2+g*(ga*a'*A*a+gi*a'*C*M*C'*a-u*norm2(A*a-B*a).^2)


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
    def __init__(self, K, A, B, M, C, y, g, ga, gi, u):
        self.K = K
        self.A = A
        self.B = B
        self.M = M
        self.C = C
        self.y = y
        self.g = g
        self.ga = ga
        self.gi = gi
        self.u = u
        assert isinstance(K, np.ndarray)
        dim = K.shape
        assert len(dim) == 2
        self.K_rows = dim[0]
        self.K_cols = dim[1]
        assert isinstance(A, np.ndarray)
        dim = A.shape
        assert len(dim) == 2
        self.A_rows = dim[0]
        self.A_cols = dim[1]
        assert isinstance(B, np.ndarray)
        dim = B.shape
        assert len(dim) == 2
        self.B_rows = dim[0]
        self.B_cols = dim[1]
        assert isinstance(M, np.ndarray)
        dim = M.shape
        assert len(dim) == 2
        self.M_rows = dim[0]
        self.M_cols = dim[1]
        assert isinstance(C, np.ndarray)
        dim = C.shape
        assert len(dim) == 2
        self.C_rows = dim[0]
        self.C_cols = dim[1]
        assert isinstance(y, np.ndarray)
        dim = y.shape
        assert len(dim) == 1
        self.y_rows = dim[0]
        self.y_cols = 1
        if isinstance(g, np.ndarray):
            dim = g.shape
            assert dim == (1, )
        self.g_rows = 1
        self.g_cols = 1
        if isinstance(ga, np.ndarray):
            dim = ga.shape
            assert dim == (1, )
        self.ga_rows = 1
        self.ga_cols = 1
        if isinstance(gi, np.ndarray):
            dim = gi.shape
            assert dim == (1, )
        self.gi_rows = 1
        self.gi_cols = 1
        if isinstance(u, np.ndarray):
            dim = u.shape
            assert dim == (1, )
        self.u_rows = 1
        self.u_cols = 1
        self.a_rows = self.B_cols
        self.a_cols = 1
        self.a_size = self.a_rows * self.a_cols
        # the following dim assertions need to hold for this problem
        assert self.y_rows == self.K_rows
        assert self.M_rows == self.M_cols == self.C_cols
        assert self.B_cols == self.a_rows == self.A_cols == self.C_rows == self.B_rows == self.A_rows == self.K_cols
        assert self.B_cols == self.a_rows == self.A_cols == self.C_rows == self.B_rows == self.A_rows == self.K_cols

    def getBounds(self):
        bounds = []
        bounds += [(-inf, inf)] * self.a_size
        return bounds

    def getStartingPoint(self):
        self.aInit = np.random.randn(self.a_rows, self.a_cols)
        return self.aInit.reshape(-1)

    def variables(self, _x):
        a = _x
        return a

    def fAndG(self, _x):
        a = self.variables(_x)
        t_0 = (self.A).dot(a)
        t_1 = ((self.K).dot(a) - self.y)
        t_2 = (self.g * self.ga)
        t_3 = (self.C.T).dot(a)
        t_4 = (self.C).dot((self.M).dot(t_3))
        t_5 = (self.g * self.gi)
        t_6 = (t_0 - (self.B).dot(a))
        t_7 = ((2 * self.g) * self.u)
        f_ = ((np.linalg.norm(t_1) ** 2) + (self.g * (((self.ga * (a).dot(t_0)) + (self.gi * (a).dot(t_4))) - (self.u * (np.linalg.norm(t_6) ** 2)))))
        g_0 = ((((((2 * (self.K.T).dot(t_1)) + (t_2 * t_0)) + (t_2 * (self.A.T).dot(a))) + (t_5 * t_4)) + (t_5 * (self.C).dot((self.M.T).dot(t_3)))) - ((t_7 * (self.A.T).dot(t_6)) - (t_7 * (self.B.T).dot(t_6))))
        g_ = g_0
        return f_, g_

def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

def solve(K, A, B, M, C, y, g, ga, gi, u):
    start = timer()
    NLP = GenoNLP(K, A, B, M, C, y, g, ga, gi, u)
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
    a = NLP.variables(x)
    solution = {}
    solution['success'] = result.success
    solution['message'] = result.message
    solution['fun'] = result.fun
    solution['grad'] = result.jac
    solution['a'] = a
    solution['elapsed'] = timer() - start
    return solution

def generateRandomData():
    np.random.seed(0)
    K = np.random.randn(3, 3)
    K = 0.5 * (K + K.T)  # make it symmetric
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    M = np.random.randn(3, 3)
    M = 0.5 * (M + M.T)  # make it symmetric
    C = np.random.randn(3, 3)
    y = np.random.randn(3)
    g = np.random.randn(1)
    ga = np.random.randn(1)
    gi = np.random.randn(1)
    u = np.random.randn(1)
    return K, A, B, M, C, y, g, ga, gi, u

if __name__ == '__main__':
    print('\ngenerating random instance')
    K, A, B, M, C, y, g, ga, gi, u = generateRandomData()
    print('solving ...')
    solution = solve(K, A, B, M, C, y, g, ga, gi, u)
    print('*'*5, 'solution', '*'*5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable a = ', solution['a'])
        print('solving took %.3f sec' % solution['elapsed'])
