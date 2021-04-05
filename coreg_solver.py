"""
Sample code automatically generated on 2021-03-28 16:51:19

by geno from www.geno-project.org

from input

parameters
  matrix K symmetric
  matrix K1 symmetric
  matrix K2 symmetric
  matrix A
  matrix B
  matrix M
  vector y
  scalar gamma
  scalar gamma1
  scalar gamma2
variables
  vector a
min
  gamma*(a'*K1*a+1/gamma2*1/gamma2*a'*A*M*A'*a+a'*B*B'*a)+norm2(y-K*a).^2


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
    def __init__(self, K, K1, K2, A, B, M, y, gamma, gamma1, gamma2):
        self.K = K
        self.K1 = K1
        self.K2 = K2
        self.A = A
        self.B = B
        self.M = M
        self.y = y
        self.gamma = gamma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        assert isinstance(K, np.ndarray)
        dim = K.shape
        assert len(dim) == 2
        self.K_rows = dim[0]
        self.K_cols = dim[1]
        assert isinstance(K1, np.ndarray)
        dim = K1.shape
        assert len(dim) == 2
        self.K1_rows = dim[0]
        self.K1_cols = dim[1]
        assert isinstance(K2, np.ndarray)
        dim = K2.shape
        assert len(dim) == 2
        self.K2_rows = dim[0]
        self.K2_cols = dim[1]
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
        assert isinstance(y, np.ndarray)
        dim = y.shape
        assert len(dim) == 1
        self.y_rows = dim[0]
        self.y_cols = 1
        if isinstance(gamma, np.ndarray):
            dim = gamma.shape
            assert dim == (1, )
        self.gamma_rows = 1
        self.gamma_cols = 1
        if isinstance(gamma1, np.ndarray):
            dim = gamma1.shape
            assert dim == (1, )
        self.gamma1_rows = 1
        self.gamma1_cols = 1
        if isinstance(gamma2, np.ndarray):
            dim = gamma2.shape
            assert dim == (1, )
        self.gamma2_rows = 1
        self.gamma2_cols = 1
        self.a_rows = self.A_rows
        self.a_cols = 1
        self.a_size = self.a_rows * self.a_cols
        # the following dim assertions need to hold for this problem
        assert self.K_rows == self.y_rows
        assert self.M_rows == self.A_cols == self.M_cols
        assert self.A_rows == self.K_cols == self.B_rows == self.K1_rows == self.K1_cols == self.a_rows

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
        t_0 = (self.K1).dot(a)
        t_1 = (self.y - (self.K).dot(a))
        t_2 = (self.gamma2 ** 2)
        t_3 = (self.A.T).dot(a)
        t_4 = (self.A).dot((self.M).dot(t_3))
        t_5 = (self.gamma / t_2)
        t_6 = (self.B).dot((self.B.T).dot(a))
        t_7 = (self.gamma * t_6)
        f_ = ((np.linalg.norm(t_1) ** 2) + (self.gamma * (((a).dot(t_0) + ((a).dot(t_4) / t_2)) + (a).dot(t_6))))
        g_0 = (((((((self.gamma * t_0) - (2 * (self.K.T).dot(t_1))) + (self.gamma * (self.K1.T).dot(a))) + (t_5 * t_4)) + (t_5 * (self.A).dot((self.M.T).dot(t_3)))) + t_7) + t_7)
        g_ = g_0
        return f_, g_

def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

def solve(K, K1, K2, A, B, M, y, gamma, gamma1, gamma2):
    start = timer()
    NLP = GenoNLP(K, K1, K2, A, B, M, y, gamma, gamma1, gamma2)
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
    K1 = np.random.randn(3, 3)
    K1 = 0.5 * (K1 + K1.T)  # make it symmetric
    K2 = np.random.randn(3, 3)
    K2 = 0.5 * (K2 + K2.T)  # make it symmetric
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    M = np.random.randn(3, 3)
    y = np.random.randn(3)
    gamma = np.random.randn(1)
    gamma1 = np.random.randn(1)
    gamma2 = np.random.randn(1)
    return K, K1, K2, A, B, M, y, gamma, gamma1, gamma2

if __name__ == '__main__':
    print('\ngenerating random instance')
    K, K1, K2, A, B, M, y, gamma, gamma1, gamma2 = generateRandomData()
    print('solving ...')
    solution = solve(K, K1, K2, A, B, M, y, gamma, gamma1, gamma2)
    print('*'*5, 'solution', '*'*5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable a = ', solution['a'])
        print('solving took %.3f sec' % solution['elapsed'])
