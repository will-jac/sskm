import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy

from manifold import construct_graph, ManifoldNorm
from kernel import SSKernelMethod

import coreg_solver_2 as coreg_solver

class SSCoRegSolver(SSKernelMethod):
    def __init__(self, kernel, g, l2, manifold, p=2, mu = 0.1, kNN = 8, weight='gaussian',
            gamma=1.0, degree=3.0, coef0=1.0, kernel_params=None, **kwargs):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.manifold_coef = manifold
        self.L2_coef = l2
        self.kNN = kNN
        self.p = p
        self.mu = mu
        self.weight = weight
        self.sd = 1.0
        self.g = g
        self.name = 'CoReg Solver'

    def fit(self, X, y, U):
        self.U_train = U
        self.X_train = X

        self.l = l = X.shape[0]
        self.u = u = U.shape[0]
        self.n = n = l + u
        N = np.append(X, U, axis=0)

        K_A = self._compute_kernel(N)
        assert(K_A.shape == (n,n))

        L, D_graph = construct_graph(N, self.kNN, self.weight, self.sd)
        D_graph = D_graph.toarray()
        D_norm = np.sqrt(np.linalg.inv(D_graph))
        M = np.matmul(np.matmul(D_norm , L.toarray()) , D_norm)
        M_norm = np.linalg.matrix_power(M, self.p) + 10**-6 * np.eye(M.shape[0])
        assert(M_norm.shape == (n,n))
        K_I = np.linalg.inv(M_norm)
        assert(K_I.shape == (n,n))

        D = 1/self.L2_coef * K_A[0:l, l:n] - 1/self.manifold_coef * K_I[0:l, l:n]
        assert(D.shape == (l, u))
        # Slighly departure from paper
        S = 1/self.L2_coef * K_A + 1/self.manifold_coef * K_I
        assert(S.shape == (n,n))
        H = np.linalg.inv(np.eye(u) + self.mu * S[l:n,l:n])
        assert(H.shape == (u,u))
        self.DH = DH = np.matmul(D , H)
        assert(DH.shape == (l, u))

        A = 1/self.manifold_coef * (K_A[0:l,0:l] + self.mu * np.matmul(DH , K_A[l:n,0:l]))
        assert(A.shape == (l,l))
        
        Bn = 1/self.manifold_coef * (K_I[0:l,:] + self.mu * np.matmul(DH , K_I[l:n,:]))
        B = Bn[0:l, 0:l] #= 1/self.manifold_coef * (K_I[0:l,0:l] + self.mu * DH @ K_I[l:n,0:l])
        assert(Bn.shape == (l,n))
        assert(B.shape == (l,l))

        K = S[0:l, 0:l] - self.mu * np.matmul(DH , D.T)
        assert(K.shape == (l,l))

        solution = coreg_solver.solve(
            K, A, B, M, Bn, y, self.g, self.L2_coef, self.manifold_coef, self.mu
        )

        print('*'*5, 'solution', '*'*5)
        print(solution['message'])
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
            np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable a = ', solution['a'])
        print('solving took %.3f sec' % solution['elapsed'])
        # print('coefs has shape', self.lm.coef_.shape)
        self.alpha = np.array(solution['a'])

    def predict(self, X):
        self.K = K = (self._compute_kernel(X, self.X_train)).T

        assert(K.shape == (self.l, X.shape[0]))
        self.K_UX = K_UX = self._compute_kernel(X, self.U_train).T
        assert(K_UX.shape == (self.u, X.shape[0]))

        # print(K.shape, K_UX.shape)
        self.K_hat = K_hat = 1/self.L2_coef * (K) #- self.mu * self.DH @ K_UX)

        # print('alpha, k_hat', self.alpha.shape, K_hat.shape)
        
        return np.dot(self.alpha, K_hat)       

class SSCoReg(SSKernelMethod):
    def __init__(self, 
            kernel, g, l2, manifold, p=2, mu = 0.1, kNN = 8, weight='gaussian',
            gamma=1.0, degree=3.0, coef0=1.0, kernel_params=None , **kwargs):
        super().__init__(kernel, gamma, degree, coef0, kernel_params)
        self.manifold_coef = manifold
        self.L2_coef = l2
        self.kNN = kNN
        self.p = p
        self.mu = mu
        self.weight = weight
        self.sd = 1.0
        self.g = g
        self.name = 'CoReg'

    def fit(self, X, y, U):
        # print('fitting coreg...')
        self.l = l = X.shape[0]
        self.u = u = U.shape[0]
        self.n = n = l + u
        # print('fitting to:', X.shape, y.shape, U.shape)
        self.U_train = U
        self.X_train = X
        N = np.append(X, U, axis=0)
        # print('l =', l, 'u =', u, 'n =', n)

        # print('computing K_A')
        # kernel 1: ambient, RLS
        K_A = self._compute_kernel(N)
        assert(K_A.shape == (n,n))

        L, D_graph = construct_graph(N, self.kNN, self.weight, self.sd)
        D_graph = D_graph.toarray()
        D_norm = np.sqrt(np.linalg.inv(D_graph))
        M = np.matmul(np.matmul(D_norm , L.toarray()) , D_norm)
        M_norm = np.linalg.matrix_power(M, self.p) + 10**-6 * np.eye(M.shape[0])
        assert(M_norm.shape == (n,n))
        K_I = np.linalg.inv(M_norm)
        assert(K_I.shape == (n,n))

        D = 1/self.L2_coef * K_A[0:l, l:n] - 1/self.manifold_coef * K_I[0:l, l:n]
        assert(D.shape == (l, u))
        # Slighly departure from paper
        S = 1/self.L2_coef * K_A + 1/self.manifold_coef * K_I
        assert(S.shape == (n,n))
        H = np.linalg.inv(np.eye(u) + self.mu * S[l:n,l:n])
        assert(H.shape == (u,u))
        self.DH = DH = np.matmul(D , H)
        assert(DH.shape == (l, u))

        A = 1/self.manifold_coef * (K_A[0:l,0:l] + self.mu * np.matmul(DH , K_A[l:n,0:l]))
        assert(A.shape == (l,l))
        
        Bn = 1/self.manifold_coef * (K_I[0:l,:] + self.mu * np.matmul(DH , K_I[l:n,:]))
        B = Bn[0:l, 0:l] #= 1/self.manifold_coef * (K_I[0:l,0:l] + self.mu * DH @ K_I[l:n,0:l])
        assert(Bn.shape == (l,n))
        assert(B.shape == (l,l))

        K = S[0:l, 0:l] - self.mu * np.matmul(DH , D.T)
        assert(K.shape == (l,l))

        self.alpha = np.linalg.solve( 
            np.matmul(K , K) + 
            self.g * np.matmul(
                self.L2_coef * A 
                + self.manifold_coef * np.matmul(np.matmul(Bn , M) , Bn.T)
                + self.mu * (
                    np.matmul(A , (A - B)) - np.matmul(B , (A - B))
                )
            ),
            np.matmul(K , y)
        )
        print(self.alpha.shape)

        # self.A = A
        # self.B = B
        # self.C = C

        # self.DH = DH

        # store what's needed for prediction
        # self.K_A = K_A[0:l,0:l]
        # self.K_I = K_I[0:l,0:l]

        # self.S_X = lambda X : 1/self.L2_coef * self._compute_kernel(X, self.K_A) + 1/self.manifold_coef * self._compute_kernel(X, self.K_I)
        # self.DH = DH
        # self.d_X = lambda X : 1/self.L2_coef * self._compute_kernel(X, self.K_A) - 1/self.manifold_coef * self._compute_kernel(X, self.K_I)

    def predict(self, X, return_extra = False):
        # print('predicting', X.shape, self.train.shape)
        # s = self.S_X(X)
        # d = self.d_X(X)
        # C = s - self.mu * self.DH @ d
        # p = 0.5 * np.dot(self.alpha.T, C)
        self.K = K = (self._compute_kernel(X, self.X_train)).T

        assert(K.shape == (self.l, X.shape[0]))
        self.K_UX = K_UX = self._compute_kernel(X, self.U_train).T
        assert(K_UX.shape == (self.u, X.shape[0]))

        # print(K.shape, K_UX.shape)
        self.K_hat = K_hat = 1/self.L2_coef * (K) #- self.mu * self.DH @ K_UX)

        # print('alpha, k_hat', self.alpha.shape, K_hat.shape)
        
        p = np.dot(self.alpha, K_hat)
        
        if return_extra:
            return p, K, K_UX, K_hat
        else:
            return p

# class SSCoMR(SSKernelMethod):
#     def __init__(self, kernel, g, l2, manifold, p=2, mu = 0.1, kNN = 8, weight='gaussian',
#             gamma=1.0, degree=3.0, coef0=1.0, kernel_params=None, **kwargs):
#         super().__init__(kernel, gamma, degree, coef0, kernel_params)
#         self.manifold_coef = manifold
#         self.L2_coef = l2
#         self.kNN = kNN
#         self.p = p
#         self.mu = mu
#         self.weight = weight
#         self.sd = 1.0
#         self.g = g
#         self.name = 'CoMR'

#     def fit(self, X, y, U):
#         print('fitting coreg...')
#         self.l = l = X.shape[0]
#         self.u = u = U.shape[0]
#         self.n = n = l + u
#         # print('fitting to:', X.shape, y.shape, U.shape)
#         self.U_train = U
#         self.X_train = X
#         N = np.append(X, U, axis=0)
#         print('l =', l, 'u =', u, 'n =', n)

#         print('computing K_A')
#         # kernel 1: ambient, RLS
#         self.K_A = K_A = self._compute_kernel(N)
#         assert(K_A.shape == (n,n))

#         print('computing K_I')
#         # kernel 2: Intrinsic, Laplacian
#         L, D_graph = construct_graph(N, self.kNN, self.weight, self.sd)

#         self.L = L
#         self.D_graph = D_graph = D_graph.toarray()
        
#         # D_graph = scipy.sparse.csr_matrix.power(scipy.sparse.linalg.inv(D_graph), 0.5)
        
#         self.D_norm = D_norm = np.sqrt(np.linalg.inv(D_graph))
#         self.M = M = D_norm @ L.toarray() @ D_norm
        
#         # I = scipy.sparse.eye(M.shape[0], format='csr')
#         # M = scipy.sparse.csr_matrix.power(M, self.p) + 10**-6 * I

#         self.M_norm = M_norm = np.linalg.matrix_power(M, self.p) + 10**-6 * np.eye(M.shape[0])
#         # M = L.toarray()
#         assert(M_norm.shape == (n,n))

#         M_norm = L.toarray()

#         # M = M.toarray()
#         self.K_I = K_I = np.linalg.pinv(M_norm)
#         assert(K_I.shape == (n,n))

#         print('computing D, S, H')
#         ## apply to both
#         self.D = D = 1/self.L2_coef * K_A[0:l, :] #- 1/self.manifold_coef * K_I[0:l, l:n]
#         # assert(D.shape == (l, u))
#         # Slighly departure from paper
#         self.S = S = 1/self.L2_coef * K_A # + 1/self.manifold_coef * K_I
#         # assert(S.shape == (n,n))
#         self.H = H = np.linalg.inv(1/self.L2_coef * K_A + 1/self.manifold_coef * K_I)
#         # assert(H.shape == (u,u))
#         self.DH = DH = D @ H
#         # assert(DH.shape == (l, u))

#         print('computing A')
#         ## back to K 2
#         # self.A = A = 1/self.manifold_coef * (K_I[0:l] + self.mu * DH @ K_I[l:n,:])
#         # assert(A.shape == (l,n))

#         self.A = A = 1/self.manifold_coef * (K_A[0:l,0:l] + self.mu * DH @ K_A[:,0:l])
#         # assert(A.shape == (l,l))
        
#         print('computing B')
#         ## apply to both
#         # self.B = B = 1/self.L2_coef * K_A[0:l, l:n] - 1/self.manifold_coef * K_I[0:l, l:n] + \
#         #     self.mu * DH @ (1/self.L2_coef * K_A[l:n, l:n] - 1/self.manifold_coef * K_I[l:n, l:n])
#         # assert(B.shape == (l,u))
#         print(DH.shape)
#         self.Bn = Bn = 1/self.manifold_coef * (K_I[0:l,:] + self.mu * DH @ K_I[:,:])
#         self.B = B = Bn[0:l, 0:l] #= 1/self.manifold_coef * (K_I[0:l,0:l] + self.mu * DH @ K_I[l:n,0:l])
#         # assert(Bn.shape == (l,n))
#         # assert(B.shape == (l,l))

        
#         # print('computing C')
#         # self.d = d = DH @ D.T
#         # assert(d.shape == (l,l))

#         # self.s_ll = s_ll = 1/self.L2_coef * K_A[0:l, 0:l] + 1/self.manifold_coef * K_I[0:l, 0:l]
#         # # print((S[0:l, 0:l]).shape, d.shape, (l,l))
#         # # print(S)
#         # # print(S[0:l,0:l])
#         # self.C = C = s_ll - self.mu * d
#         # assert(C.shape == (l,l))

#         print('computing K')
#         self.K = K = S[0:l, 0:l] - self.mu * DH @ D.T
#         # assert(K.shape == (l,l))

#         print('solving')
#         #  
#         # self.alpha = 0.5 * np.linalg.inv(
#         #     ### f norm
#         #     2 * self.g * (
#         #         # f_A norm
#         #         self.L2_coef * np.eye(l) @ K_A[0:l, 0:l] + 
#         #         # f_I norm
#         #         self.manifold_coef * A @ M @ A.T # + 
#         #         # diff norm
#         #         # self.mu * B @ B.T
#         #     )
#         #     ### target
#         #     # + (1/4) * C @ C.T
#         #     # + K_A[0:l, 0:l]
#         #     + np.eye(l) @ K_A[0:l,0:l]
#         # ) @ y #@ C @ y

#         self.alpha = np.linalg.solve( 
#             K @ K + 
#             self.g * (
#                 self.L2_coef * A 
#                 + self.manifold_coef * Bn @ M @ Bn.T
#                 + self.mu * (
#                     A @ (A - B) - B @ (A - B)
#                 )
#             ),
#             K @ y
#         )
#         print(self.alpha.shape)

#     def predict(self, X, return_extra = False):
#         self.K = K = (self._compute_kernel(X, self.X_train)).T

#         # assert(K.shape == (self.l, X.shape[0]))
#         self.K_UX = K_UX = self._compute_kernel(X, np.append(self.X_train, self.U_train, axis=0)).T
#         # assert(K_UX.shape == (self.u, X.shape[0]))

#         print(K.shape, K_UX.shape)
#         self.K_hat = K_hat = 1/self.L2_coef * (K - self.mu * self.DH @ K_UX)

#         print('alpha, k_hat', self.alpha.shape, K_hat.shape)
        
#         p = np.dot(self.alpha, K_hat)
        
#         if return_extra:
#             return p, K, K_UX, K_hat
#         else:
#             return p