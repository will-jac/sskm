import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

## Helper kernels

def linear():
    return lambda x, y : np.inner(x,y)

def gaussian(sigma=1.0):
    return lambda x, y : np.exp(-np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2)))

def hyperbolic_tangent(kappa, c):
    return lambda x, y : np.tanh(kappa * np.dot(x, y) + c)

def radial_basis(gamma=10):
    return lambda x, y : np.exp(-gamma * np.linalg.norm(np.subtract(x, y)))

# Replaced by sklearn pairwise
def compute_kernel_matrix(kernel, X, K=None):
    print('constructing gram matrix for X:', X.shape)
    n_samples, _ = X.shape
    K = np.zeros((n_samples, n_samples))

    # TODO: vectorize / cython this
    for i, x_i in enumerate(X):
        for j, x_j in enumerate(X):
            K[i, j] = kernel(x_i, x_j)

    print('K constructed:', K.shape)
    return K

from sklearn.metrics.pairwise import pairwise_kernels

class KernelMethod():
    '''
    Base class for kernel methods, which solve an optimization problem of the following form:
    min_{f in H_K} 1/l sum(loss(y_i, f(x_i))) + gamma ||f||^2_K

    which has the general solution
    f* = sum(alpha_i * K(x, x_i))
    '''
    def __init__(self, kernel, gamma=None, degree=3.0, coef0=1.0, kernel_params=None, **kwargs):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _compute_kernel(self, X, K=None):
        if callable(self.kernel):
            params = self.kernel_params
        elif self.kernel == 'precomputed':
            return X
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, K, metric=self.kernel, filter_params=True, **params)

        # if callable(self._kernel):
        #     if K is None:
        #         return compute_kernel_matrix(self._kernel, X)
        #     else:
        #         return compute_kernel_matrix(self.kernel, X, K)

        # elif self._kernel == 'precomputed':
        #     # X is already a kernelized matrix
        #     return X
        # else:
        #     print('warning: no suitable kernel found. Cannot continue.')
        #     return

    def fit(self, X, y, U=None):
        K = self._compute_kernel(X)
        # print('computing kernel for X', X.shape, 'K has shape', K.shape)
        self.X_train = X
        self._solve(K, y)

    def predict(self, X):
        return

    def _solve(self, K, y):
        return

class SSKernelMethod(KernelMethod):
    '''
    Base class for semi-supervised kernel methods, which solve an optimization problem of the following form:
    min_{f in H_K} 1/l sum(loss(y_i, f(x_i))) + gamma ||f||^2_K

    which has the general solution
    f* = sum(alpha_i * K(x, x_i))
    '''
    def __init__(self, kernel, gamma=None, degree=3.0, coef0=1.0, kernel_params=None, **kwargs):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)

    def fit(self, X, y, U):
        # ...
        return
    def predict(self, X):
        # ...
        return
    def _solve(self, K, y):
        # ...
        return

# standard least-squares regression w/ Ridge regression (eg L2 regularization)
from sklearn.linear_model import Ridge as _Ridge

class RidgeKernel(KernelMethod):

    def __init__(self, kernel, L2_coef, gamma=None, degree=3, coef0=1, kernel_params=None, **kwargs):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.lm = _Ridge(alpha = L2_coef)
        self.name = 'ridge'

    def _solve(self, K, y):
        # print('fitting lm to K, y:', K.shape, y.shape)
        self.lm.fit(K, y)
        # print('coefs has shape', self.lm.coef_.shape)

    def predict(self, X):
        # print('predicting X with shape', X.shape)
        K = self._compute_kernel(X, self.X_train)
        # print('X in the kernel space has shape', K.shape)
        # return np.dot(K, self.dual_coef_)
        p = self.lm.predict(K)
        # print(p.shape, p)
        return p

class RLSKernel(KernelMethod):
    '''
    Solution leads to the form:

    alpha = (K - gamma * l * I)^-1 * Y
    '''
    def __init__(self, kernel, gamma, simple=True, solve=False, **kwargs):
        super().__init__(kernel, gamma)
        self.name = 'RLS'
        self.simple = simple
        self.solve = solve

    def _solve(self, K, y):
        print(K.shape)
        if self.solve:
            if self.simple:
                self.alpha = np.linalg.solve(K + 1* np.eye(K.shape[0]), y)
            # else:
            #     self.alpha = np.linalg.solve(K @ K + K, K @ y)
        # else:
        #     if self.simple:
        #         self.alpha = np.linalg.inv(K + 1* np.eye(K.shape[0])) @ y
            # else:
            #     self.alpha = np.linalg.inv(K @ K + K) @ K @ y

        # self.alpha = np.linalg.inv(
        #     K @ K + np.eye(K.shape[0])
        # ) @ K @ y
    def predict(self, X):
        # TODO: vectorize
        # XX = self.X * X.T
        # test_norms = (np.multiply(X.T, X.T).sum(axis=0)).T
        # K = np.array(np.ones((m, 1), dtype = np.float64)) * test_norms.T
        # K = K + self.K * np.array(np.ones((1, n), dtype = np.float64))
        # K = K - 2 * XX
        # K = - gamma * K
        # K = np.exp(K)
        # return K.A.T
        K = self._compute_kernel(X, self.X_train)
        return np.dot(self.alpha, K.T)

import least_squares as ls

class LS():
    def __init__(self, **kwargs):
        self.name = 'least squares'

    def fit(self, X, y, U=None):
        solution = ls.solve(X, y)
        print('*'*5, 'solution', '*'*5)
        print(solution['message'])
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
            np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable x = ', solution['x'])
        print('solving took %.3f sec' % solution['elapsed'])
        self.alpha = np.array(solution['x'])

    def predict(self, X):
        p = np.dot(X, self.alpha)
        print('predicting X', X.shape, '->', p.shape)
        return p

class KLS(KernelMethod):
    def __init__(self, kernel, gamma=None, degree=3, coef0=1, kernel_params=None, **kwargs):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.name = 'kernel least squares'

    def _solve(self, K, y):
        # print('fitting lm to K, y:', K.shape, y.shape)
        solution = ls.solve(K, y)
        print('*'*5, 'solution', '*'*5)
        print(solution['message'])
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
            np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable x = ', solution['x'])
        print('solving took %.3f sec' % solution['elapsed'])
        # print('coefs has shape', self.lm.coef_.shape)
        self.alpha = np.array(solution['x'])

    def predict(self, X):
        # print('predicting X with shape', X.shape)
        K = self._compute_kernel(X, self.X_train)
        # print('X in the kernel space has shape', K.shape)
        # return np.dot(K, self.dual_coef_)
        p = np.dot(K, self.alpha)
        print('predicting X', X.shape, '->', K.shape, '->', p.shape)

        # print(p.shape, p)
        return p

