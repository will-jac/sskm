import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy


def construct_graph(X, k=8, weight='binary', sd=1.0):
    if weight == 'binary':
        weight = lambda x1, x2 : 1
    elif weight == 'gaussian':
        weight = lambda x1, x2 : np.exp(-(x1 - x2)**2 / (2*(sd**2)))

    G = pairwise_distances(X, metric='euclidean')
    # print('G:')
    # print(G)

    # k+1 because self is always closest
    neighbors = np.argpartition(G, k+1)[:,:(k+1)]
    # print(neighbors)

    # I'm sure there's a faster way to do this
    W = np.zeros(G.shape)
    for i, row in enumerate(neighbors):
        for n in row:
            W[i][n] = weight(G[i][i], G[i][n])
        # remove the self connection
        W[i,i] = 0

    # this array maybe should be symmetric? but I don't think so
    # this isn't *strictly* an adjacency

    # print('W:')
    # print(W)
    
    # make it sparse
    W = scipy.sparse.csr_matrix(W)

    # return the laplacian
    D = scipy.sparse.diags([k for _ in range(W.shape[0])], format='csr')
    return scipy.subtract(D, W), D

class ManifoldNorm():
    def __init__(self, k=10, weight='gaussian', sd=1):
        self.k = k
        self.weight = weight
        self.sd = float(sd)

    def norm(self, X, X_pred, ):
        L, _ = construct_graph(X, self.k, self.weight, self.sd)
        
        return scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(X_pred.T, L), X_pred)

from kernel import KernelMethod

class ManifoldRLS(KernelMethod):
    def __init__(self, kernel, manifold_coef, kNN = 8, weight='gaussian',
            gamma=None, degree=3, coef0=1, kernel_params=None):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.manifold_coef = manifold_coef
        self.kNN = kNN
        self.weight = weight
        self.sd = 1.0
        self.name = 'Manifold RLS'

    def _solve(self, K, y):
        L, _ = construct_graph(self.X_train, self.kNN, self.weight, self.sd)
        # print('solving: K, L, y', K.shape, L.shape, y.shape)
        self.alpha = np.dot(np.linalg.pinv(
                np.eye(K.shape[0]) @ K 
                + self.manifold_coef * (L @ K)
            ), y)

    def predict(self, X):
        K = self._compute_kernel(X, self.X_train)
        # print('predicting', self.alpha.shape, K.shape)
        p = np.dot(self.alpha, K.T)
        # print(p.shape, p)
        return p

class LapRLS(KernelMethod):
    def __init__(self, kernel, L2_coef, manifold_coef, kNN = 8, weight='gaussian',
            gamma=None, degree=3, coef0=1, kernel_params=None):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.L2_coef = L2_coef
        self.manifold_coef = manifold_coef
        self.kNN = kNN
        self.weight = weight
        self.sd = 1.0
        self.name = 'Laplacian RLS'
        
    def _solve(self, K, y):
        L, _ = construct_graph(self.X_train, self.kNN, self.weight, self.sd)
        # print('solving: K, L, y', K.shape, L.shape, y.shape)
        self.alpha = np.dot(np.linalg.pinv(
            np.eye(K.shape[0]) @ K 
                + self.L2_coef * np.eye(K.shape[0])
                + self.manifold_coef * (L @ K)
            ), y)

    def predict(self, X):
        K = self._compute_kernel(X, self.X_train)
        # print('predicting:', self.alpha.shape, K.shape)
        p = np.dot(self.alpha, K.T)
        # print(p.shape, p)
        return p

if __name__ == '__main__':
    X = np.array([[0,0],[0,1],[1,1],[1,2],[2,2]])#,[2,3],[3,3],[3,4],[4,4]])
    y = np.array([[0],[1],[1],[2],[2]])
    L, _ = construct_graph(X, 2, weight='gaussian')
    print(L.toarray())
    mn = ManifoldNorm(k=2)
    n = mn.norm(X, y)
    print(n)