import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy

from manifold import construct_graph, ManifoldNorm
from ss_kernel import SSKernelMethod

class SSManifoldRLS(SSKernelMethod):
    def __init__(self, kernel, manifold_coef, kNN = 8, weight='gaussian',
            gamma=None, degree=3, coef0=1, kernel_params=None):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.manifold_coef = manifold_coef
        self.kNN = kNN
        self.weight = weight
        self.sd = 1.0
        self.name = 'Manifold RLS'

    def fit(self, X, y, U):
        self.l = X.shape[0]
        self.u = U.shape[0]
        # print('fitting to:', X.shape, y.shape, U.shape)
        self.train = np.append(X, U, axis=0)
        # print('train has shape:', self.train.shape)
        K = self._compute_kernel(self.train)
        self._solve(K, y)

    def _solve(self, K, y):
        J = scipy.sparse.diags(
            [1 for _ in range(self.l)] + [0 for _ in range(self.u)],
            format='csr')
        y_n = np.append(y, np.zeros((self.u)))
        L, _ = construct_graph(self.train, self.kNN, self.weight, self.sd)

        # print('solving:', J.shape, K.shape, L.shape, y_n.shape)
        sol = scipy.sparse.linalg.lsmr(
            J @ K + self.manifold_coef * (L @ K),
            y_n
        )
        self.alpha = sol[0]
        print('alpha found!', sol[1], self.alpha.shape)
    
    def predict(self, X):
        # print('predicting', X.shape, self.train.shape)
        K = self._compute_kernel(X, self.train)
        p = np.dot(self.alpha, K.T)
        return p

class SSLapRLS(SSKernelMethod):
    def __init__(self, kernel, L2_coef, manifold_coef, kNN = 8, weight='gaussian',
            gamma=None, degree=3, coef0=1, kernel_params=None):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.manifold_coef = manifold_coef
        self.L2_coef = L2_coef
        self.kNN = kNN
        self.weight = weight
        self.sd = 1.0
        self.name = 'Lap RLS'

    def fit(self, X, y, U):
        self.l = X.shape[0]
        self.u = U.shape[0]
        # print('fitting to:', X.shape, y.shape, U.shape)
        self.train = np.append(X, U, axis=0)
        K = self._compute_kernel(self.train)
        self._solve(K, y)

    def _solve(self, K, y):
        L, _ = construct_graph(self.train, self.kNN, self.weight, self.sd)
        J = scipy.sparse.diags(
            [1 for _ in range(self.l)] + [0 for _ in range(self.u)],
            format='csr')
        I = scipy.sparse.eye(K.shape[0], format='csr')
        y_n = np.append(y, np.zeros((self.u)))
        
        sol = scipy.sparse.linalg.lsmr(
            (
                J @ K 
                + self.manifold_coef * (L @ K) 
                + self.L2_coef * I
            ), y_n
        )
        self.alpha = sol[0]
        print('alpha found!', sol[1], self.alpha.shape)
        # print(self.alpha)

    def predict(self, X):
        # print('predicting', X.shape, self.train.shape)
        K = self._compute_kernel(X, self.train)
        p = np.dot(self.alpha, K.T)
        return p