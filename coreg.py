import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy

from manifold import construct_graph, ManifoldNorm
from ss_kernel import SSKernelMethod

class SSCoReg(SSKernelMethod):
    def __init__(self, kernel, g, L2_coef, manifold_coef, p=2, mu = 0.1, kNN = 8, weight='gaussian',
            gamma=None, degree=3, coef0=1, kernel_params=None):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.manifold_coef = manifold_coef
        self.L2_coef = L2_coef
        self.kNN = kNN
        self.p = p
        self.mu = mu
        self.weight = weight
        self.sd = 1.0
        self.g = g
        self.name = 'CoReg'

    def fit(self, X, y, U):
        print('fitting coreg...')
        self.l = l = X.shape[0]
        self.u = u = U.shape[0]
        n = l + u
        # print('fitting to:', X.shape, y.shape, U.shape)
        self.U_train = U
        self.X_train = X
        print('l =', l, 'u =', u, 'n =', n)

        # print('computing K_A')
        # kernel 1: ambient, RLS
        K_A = self._compute_kernel(np.append(X, U, axis=0))
        assert(K_A.shape == (n,n))

        # print('computing K_I')
        # kernel 2: Intrinsic, Laplacian
        L, D_graph = construct_graph(np.append(X, U, axis=0), self.kNN, self.weight, self.sd)
        D_graph = D_graph.toarray()
        # D_graph = scipy.sparse.csr_matrix.power(scipy.sparse.linalg.inv(D_graph), 0.5)
        np.sqrt(np.linalg.inv(D_graph), out=D_graph)
        M = D_graph @ L @ D_graph
        # I = scipy.sparse.eye(M.shape[0], format='csr')
        # M = scipy.sparse.csr_matrix.power(M, self.p) + 10**-6 * I
        M = np.linalg.matrix_power(M, self.p) * 10**-6 * np.eye(M.shape[0])
        assert(M.shape == (n,n))

        # M = M.toarray()
        K_I = np.linalg.inv(M)
        assert(K_I.shape == (n,n))

        # print('computing D, S, H')
        ## apply to both
        D = 1/self.L2_coef * K_A[0:l, l:n] - 1/self.manifold_coef * K_I[0:l, l:n]
        assert(D.shape == (l, u))
        S = 1/self.L2_coef * K_A[l:n, l:n] + 1/self.manifold_coef * K_I[l:n, l:n]
        assert(S.shape == (u,u))
        H = np.linalg.inv(np.eye(S.shape[0]) + self.mu * S)
        assert(H.shape == (u,u))
        DH = D @ H
        assert(DH.shape == (l, u))

        # print('computing A')
        ## back to K 2
        A = 1 / self.manifold_coef * (K_I[0:l] + self.mu * DH @ K_I[l:n,:])
        assert(A.shape == (l,n))

        # print('computing B')
        ## apply to both
        B = 1/self.L2_coef * K_A[0:l, l:n] - 1/self.manifold_coef * K_I[0:l, l:n] + \
            self.mu * DH @ (1/self.L2_coef * K_A[l:n, l:n] - 1/self.manifold_coef * K_I[l:n, l:n])
        assert(B.shape == (l,u))

        # print('computing C')
        d = DH @ D.T
        assert(d.shape == (l,l))

        s_ll = 1/self.L2_coef * K_A[0:l, 0:l] + 1/self.manifold_coef * K_I[0:l, 0:l]
        # print((S[0:l, 0:l]).shape, d.shape, (l,l))
        # print(S)
        # print(S[0:l,0:l])
        C = s_ll - self.mu * d
        assert(C.shape == (l,l))

        # print('solving')
        sol = 0.5 * np.linalg.inv(
            ### f norm
            2 * self.g * (
                # f_A norm
                K_A[0:l, 0:l] + 
                # f_I norm
                A @ M @ A.T + 
                # diff norm
                B @ B.T
            ) +
            ### target
            (1/4) * C @ C.T
        ) @ C @ y

        print(sol.shape)
        self.alpha = sol
        self.A = A
        self.B = B
        self.C = C

        self.DH = DH

        # store what's needed for prediction
        # self.K_A = K_A[0:l,0:l]
        # self.K_I = K_I[0:l,0:l]

        # self.S_X = lambda X : 1/self.L2_coef * self._compute_kernel(X, self.K_A) + 1/self.manifold_coef * self._compute_kernel(X, self.K_I)
        # self.DH = DH
        # self.d_X = lambda X : 1/self.L2_coef * self._compute_kernel(X, self.K_A) - 1/self.manifold_coef * self._compute_kernel(X, self.K_I)

    def predict(self, X):
        # print('predicting', X.shape, self.train.shape)
        # s = self.S_X(X)
        # d = self.d_X(X)
        # C = s - self.mu * self.DH @ d
        # p = 0.5 * np.dot(self.alpha.T, C)
        K = self._compute_kernel(X, self.X_train)
        K_UX = self._compute_kernel(X, self.U_train)
        print(K.shape, K_UX.shape)
        K_hat = 1/self.L2_coef * (K.T - self.mu * self.DH @ K_UX.T)
        print(self.alpha.shape, K_hat.shape)
        p = self.alpha.T @ K_hat
        return p