import numpy as np

from kernel import RidgeKernel
from sklearn.metrics.pairwise import pairwise_kernels

from scipy.linalg import svd

rng = np.random.default_rng()    

class NystromTransformer():
    # https://www.stat.berkeley.edu/~mmahoney/pubs/kernel_JMLR.pdf

    def __init__(self, kernel, n_components = 100, gamma=None, degree=None, coef0=None, 
            kernel_params=None, n_jobs=None, **kwargs):
        self.kernel = kernel
        self.n_components = n_components
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def _compute_kernel(self, X, K=None):
        if callable(self.kernel):
            params = self.kernel_params
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, K, metric=self.kernel, filter_params=True, 
                n_jobs=self.n_jobs, **params)

    def fit3(self, X, y):
        # make sure that the shapes will work
        if X.shape[0] < self.n_components:
            print('warning: more components than samples, rescaling to number of samples')
            n_components = X.shape[0]
        else:
            n_components = self.n_components

        # TODO: probability sampling
        samp = rng.permutation(X.shape[0])
        samp = samp[0:n_components]
        # X basis 
        X_hat = X[samp]

        A = self._compute_kernel(X, X_hat)
        B = A[samp, samp]   

    def fit2(self, X, y):
        # https://www.stat.berkeley.edu/~mmahoney/pubs/kernel_JMLR.pdf
        
        # make sure that the shapes will work
        if X.shape[0] < self.n_components:
            print('warning: more components than samples, rescaling to number of samples')
            n_components = X.shape[0]
        else:
            n_components = self.n_components

        # TODO: probability sampling
        samp = rng.permutation(X.shape[0])
        samp = samp[0:n_components]
        # X basis 
        self.X_hat = X[samp]

        self.C = []
        for xhat in self.X_hat:
            c = []
            for x in X:
                c.append(self.kernel(x, xhat))
            self.C.append(c)
        self.C = np.array(self.C)
        print('shape of C:', self.C.shape)
        
        print('alternative method:', self._compute_kernel(X, self.X_hat).shape)

        # TODO: scaling

    def fit(self, X, y):
        # make sure that the shapes will work
        if X.shape[0] < self.n_components:
            print('warning: more components than samples, rescaling to number of samples')
            n_components = X.shape[0]
        else:
            n_components = self.n_components
        # first, sample columns of the kernel matrix K

        # sample m rows of the data
        samp = rng.permutation(X.shape[0])
        samp = samp[0:n_components]
        # X basis 
        self.X_hat = X[samp]

        # compute the kernel matrix for our basis kernel
        K_hat = self._compute_kernel(self.X_hat)
        # print('computed K_hat with shape', K_hat.shape)

        # K_b = self._compute_kernel(X, self.X_hat)
        # K_r = K_b @ np.linalg.inv(K_hat) @ K_b.T

        # print('computing svd of K_r', K_r.shape)
        U, S, V = svd(K_hat)

        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = self.X_hat
        self.component_indices_ = samp
        # print('normalization has shape', self.normalization_.shape)
        # print('components has shape', self.X_hat.shape)
        
        # 1 / D creates the inverse, only true because it's diagonal
        # d = d[0:n_components]
        # D_inverse = np.diag(np.sqrt(1 / np.maximum(d, 1e-12)))

        # VtL = Vt[0:n_components,:]

        # # print(D_inverse)
        # print('computing Z from D, V', D_inverse.shape, VtL.shape)
        # self.Z = D_inverse @ VtL
        # print('Z has shape', self.Z.shape)
        return self
    
    def transform(self, X):
        # print('transforming X with shape', X.shape)
        # kernel_space_X = self._compute_kernel(X, self.X_hat)
        kernel_space_X = self._compute_kernel(X, self.X_hat)
        # print('X in the kernel space has shape', kernel_space_X.shape)
        to_ret = np.dot(kernel_space_X, self.normalization_.T)
        # print('normalized X has shape', to_ret.shape)
        return to_ret
        # z = self.Z.T @ kernel_space_X.T
        # print('z(X) has shape', z.shape)
        # return z.T
