import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy

def construct_graph(X, k=8, weight='binary', sd=1.0, plot_manifold=False):
    G = pairwise_distances(X, metric='euclidean')
    
    if weight == 'binary':
        weight = lambda i, j : 1
    elif weight == 'gaussian':
        weight = lambda i, j : np.exp(-np.sum((X[i] - X[j])**2 / (2*(sd**2))))
    elif weight == 'euclidean':
        weight = lambda i, j : G[i,j]

    # k+1 because self is always closest
    neighbors = np.argpartition(G, k+1)[:,:(k+1)]
    # print(neighbors)

    # I'm sure there's a faster way to do this
    W = np.zeros(G.shape)
    for i, row in enumerate(neighbors):
        for n in row:
            W[i,n] = weight(i, n)
            # make it symmetric (this breaks the requirement of exactly k NN, instead at least k NN)
            W[n,i] = W[i,n]
        # remove the self connection
        W[i,i] = 0
    
    # make it sparse
    W = scipy.sparse.csr_matrix(W)

    if plot_manifold:
        import matplotlib.pyplot as plt
        plt.title(f'Manifold graph')
        plt.scatter(X[:,0], X[:,1])

        (rows, cols) = W.nonzero()
        for i, j in zip(rows, cols):
            x = [X[i,0], X[j,0]]
            y = [X[i,1], X[j,1]]
            plt.plot(x,y)
        plt.show()

    # return the laplacian
    D = scipy.sparse.diags([k for _ in range(W.shape[0])], format='csr')
    return (W - D), D

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
    def __init__(self, kernel, manifold, kNN = 8, weight='gaussian',
            gamma=None, degree=3, coef0=1, kernel_params=None):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.manifold_coef = manifold
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
    def __init__(self, kernel, l2, manifold, kNN = 8, weight='gaussian',
            gamma=None, degree=3, coef0=1, kernel_params=None):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.L2_coef = l2
        self.manifold_coef = manifold
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


# Y = np.array([[0,0],[1,1],[2,2],[3,3],[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]])
# L, D = construct_graph(Y, k=3)

if __name__ == '__main__':
    X = np.array([[0,0],[0,1],[1,1],[1,2],[2,2]])#,[2,3],[3,3],[3,4],[4,4]])
    y = np.array([[0],[1],[1],[2],[2]])
    L, _ = construct_graph(X, 2, weight='gaussian')
    print(L.toarray())
    mn = ManifoldNorm(k=2)
    n = mn.norm(X, y)
    print(n)