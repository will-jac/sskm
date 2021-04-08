# Standard SVM code

# THIS IS NOT PERFORMANT, DO NOT USE

import numpy as np
import cvxopt.solvers

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SVM:

    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def fit(self, X, y, **kwargs):
        lagrange_mult = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_mult)

    def _compute_multipliers(self, X, y):
        n_samples = X.shape[0]

        K = self._gram_matrix(X)
        
        # solve
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b 

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.ones(n_samples))

        # a_i \leq 0
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        print('solving...')
        print('P', P.size, 'q', q.size, 'G', G.size, 'h', h.size, 'A', A.size, 'b', b.size)
        solution = cvxopt.solvers.qp(P, q, G, h) #, A, b)

        return np.ravel(solution['x'])

    def _gram_matrix(self, X):
        print('constructing gram matrix for X:', X.shape)
        n_samples, _ = X.shape
        K = np.zeros((n_samples, n_samples))

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        
        print('K constructed:', K.shape)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        self._bias = 0.0
        self._weights = lagrange_multipliers[support_vector_indices]
        self._vectors = X[support_vector_indices]
        self._labels = y[support_vector_indices]

        self._bias = np.mean([y_k - self._predict_sample(x_k) for (y_k, x_k) in zip(self._labels, self._vectors)])

        return

    def _predict_sample(self, x):
        result = self._bias
        for z_i, x_i, y_i, in zip(self._weights, self._vectors, self._labels):
            result += z_i * y_i * self._kernel(x_i, x)
        # print(result)
        return np.sign(result).item()

    def predict(self, X):
        labels = []
        for x in X:
            # print(x)
            labels.append(self._predict_sample(x))
        l = np.array(labels)
        print(l)
        return l