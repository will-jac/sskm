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

class SSKernelMethod():
    '''
    Base class for semi-supervised kernel methods, which solve an optimization problem of the following form:
    min_{f in H_K} 1/l sum(loss(y_i, f(x_i))) + gamma ||f||^2_K

    which has the general solution
    f* = sum(alpha_i * K(x, x_i))
    '''
    def __init__(self, kernel, gamma=None, degree=3.0, coef0=1.0, kernel_params=None):
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

    def fit(self, X, y, U):
        ...
        
    def predict(self, X):
        ...

    def _solve(self, K, y):
        ...

