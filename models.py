# others
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# mine
from random_fourier_features import rff
from nystrom import NystromTransformer
from manifold import ManifoldRLS, LapRLS
from ss_manifold import SSManifoldRLS, SSLapRLS
from coreg import SSCoReg, SSCoMR, SSCoRegSolver
import kernel

from SVM import SVM

import util

svm = SVC()
svm.name = 'SVM'

from sklearn.kernel_approximation import Nystroem

class Nystrom_Ridge():
    def __init__(self, n_components=1000):
        self.name = 'Nystrom + Ridge, m='+str(n_components)
        self.transformer =  NystromTransformer('rbf', n_components=n_components, gamma=0.2)
        # self.transformer = Nystroem(gamma=0.2, n_components=10)
        self.kernel_method = kernel.RidgeKernel('precomputed', 1.0)
    def fit(self, X, y):
        X = self.transformer.fit(X, y).transform(X)
        self.kernel_method.fit(X, y)

    def predict(self, X):
        X = self.transformer.transform(X)
        return self.kernel_method.predict(X)


models = {
    'rff': rff,
    'svm': svm,
    'Nystrom_Ridge': Nystrom_Ridge,
    'LS': kernel.LS,
    'KLS': kernel.KLS,
    'RLSKernel': kernel.RLSKernel,
    'RidgeKernel': kernel.RidgeKernel,
    'ManifoldRLS': ManifoldRLS,
    'SSManifoldRLS': SSManifoldRLS,
    'SSLapRLS': SSLapRLS,
    'SSCoMR': SSCoMR,
    'SSCoReg': SSCoReg,
    'SSCoRegSolver': SSCoRegSolver,
}