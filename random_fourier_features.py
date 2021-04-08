# Random Features for Large-Scale Kernel Machines
# https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

# Will be implementing the Random Fourier Features described above

# algorithm:
# Compute the fourier transform p of the kernel k: p(w) = 1/(2pi) \int e^{-jw'\delta}k(\delta) d\Delta
# Draw D iid samples w1 ... wD in Rd from p and D iid samples b1 .. bD in R from [0, 2pi]
# z(x) = \sqrt(2/D) [cos(w1' x b1) ... cos(wD' x + bD)]'

import numpy as np
from numpy import random

# standard least-squares regression
from sklearn.linear_model import Ridge
# allows co-variance computation
from scipy.linalg import cholesky, cho_solve

# transformations possible

def gaussian(w, D):
    return (2*np.pi)**(-D/2) * np.exp(np.linalg.norm(w)**2 / (-2))

def laplacian(w, D):
    transf = lambda w : 1/(np.pi*(1+w**2))
    return np.prod(np.apply_along_axis(transf, axis=0, arr=w))

def cauchy(w, D):
    return -1 # TODO

rng = np.random.default_rng()    

class rff:

    def __init__(self, D = 10, sigma = 1.0, alpha=1.0, k = 'gaussian', method = 'linear', **kwargs):
        """Random Fourier Featurues
        D : Dimension of random features
        k : kernel to use (`gaussian`, `laplacian`)
        method : method used to solve linear regression (`linear`, `cholesky`)
        """
        self.name = 'rff, D='+str(D)
        if k == 'laplacian':
            self.p = laplacian
        elif k == 'cauchy':
            self.p = cauchy
        else:
            self.p = gaussian # default

        if method == 'cholesky':
            self._fit = self._fit_cholesky
        else:
            self._fit = self._fit_linear
            self.lm = Ridge(alpha=alpha)

        self.sigma = sigma
        self.D = D
        self.b = None
        self.W = None

    def _fit_linear(self, y):
        self.lm.fit(self.Z.T, y)
        
    def _fit_cholesky(self, y):

        sigma_I = self.sigma * np.eye(self.N)
        self.K = self.Z.T @ self.Z + sigma_I

    def fit(self, X, y):
        """Fit model with data X and labels y
        """
        # compute W, b
        # pull W from the normal distribution, and b from 0->2 pi

        self.N, d = X.shape

        # weights go from R^d -> R^D
        self.W = rng.normal(loc=0, scale=1, size=(self.D, d))
        # bias is in R, need D terms
        self.b = rng.uniform(0, 2*np.pi, size=self.D)

        self.Z = self.compute_features(X)
        
        self._fit(y)
        self.fitted = True

        # now solve the least-squares problem:
        # min_w ||Z'w - y||_2^2 + \lambda ||w||_2^2

        # done via linear equation solver, eg:
        # A x = b to solve x
        # use cholesky solver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_solve.html

        # self.L = cholesky(self.kernel, lower=True)
        # self.alpha = cho_solve((self.L, True), y)

    def predict(self, X):
        # if self.alpha is None:
        #     print('error: model not fitted')
        #     return
        
        # Z = self.compute_features(X)
        # K = Z.T @ self.Z
        # y_mean = K.dot(self.alpha)

        # v = cho_solve((self.L, True), K.T)
        # y_cov = (Z.T @ Z) - K.dot(v)

        # return y_mean, y_cov

        Z = self.compute_features(X)
        return self.lm.predict(Z.T)

    def compute_features(self, X):
        # TODO: tweak as described https://teddykoker.com/2020/11/performers/
        N, d = X.shape
            
        B = np.repeat(self.b[:, np.newaxis], N, axis=1)

        # @ = matix multiplication
        # print('W:', self.W.shape, 'X.T:', X.T.shape)
        Z = 1.0/np.sqrt(2.0*self.D) * np.cos(self.sigma * self.W @ X.T + B)

        return Z
