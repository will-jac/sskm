import numpy as np
# import matplotlib.pyplot as plt

import argparse

# ML methods
# others
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# mine
from random_fourier_features import rff
from nystrom import NystromTransformer
from manifold import ManifoldRLS, LapRLS

import kernel
from SVM import SVM

import util

# data sets
# import data.adult as adult
import data.sphere as sphere
import data.checkerboard as checkerboard

def test_runner(models, test_funcs, test_funcs_names):
    for test_f, test_name in zip(test_funcs, test_funcs_names):
        print('---- running test:', test_name, ' ----')
        for model in models:
            test_f(model)

def sphere_test(model, n=10, u=1000, d = 1000, show_plots=False):

    X, y, U = sphere.generate_data(n=n, d=d, u=u)
    (train, test) = util.train_test_valid_split(X, y, split=(0.7, 0.3), U=U)

    # if show_plots:
    #     f = plt.figure()
    #     plt.title('Labeled Data (input)')
    #     labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y]
    #     plt.scatter(X[:,0], X[:,1], c=labels)
    #     plt.show()
    #     plt.close(f)

    # for model, name in zip(models, model_names):
    model.fit(train.X, train.y, train.U)

    y_pred = model.predict(test.X)
    y_pred = y_pred.ravel()
    
    # classify
    for i, y_p in enumerate(y_pred):
        if y_p > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    wrong = util.percent_wrong(y_pred.ravel(), test.y.ravel())
    acc = 1.0 - wrong
    print(model.name, ' : acc:', acc)

    # if show_plots:
    #     plt.figure()
    #     plt.title(f'{model.name} on sphere (d={d}), %wrong={wrong}')
    #     # plot (for sphere)
    #     labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y_pred]
    #     plt.scatter(test.X[:,0], test.X[:,1], c=labels)
    #     plt.show()
    #     plt.close(f)

def checkerboard_test(model, shape=(1000, 2), noise=0.1, seed=None):
    X, y = checkerboard.generate_data(shape=shape, noise=noise, seed=seed, shuffle=True)
    
    (train, test) = util.train_test_valid_split(X, y, split=(0.7, 0.3))

    model.fit(train.X, train.y)

    y_pred = model.predict(test.X)
    y_pred = y_pred.ravel()
    
    # classify
    for i, y_p in enumerate(y_pred):
        if y_p > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    wrong = util.percent_wrong(y_pred.ravel(), test.y.ravel())
    acc = 1.0 - wrong
    print(model.name, ' : acc:', acc)

    # labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y_pred]
    # plt.scatter(test.X[:,0], test.X[:,1], c=labels)
    # plt.show()

def adult_test(model):

    (train, test) = util.train_test_valid_split(adult.X, adult.y, split=(0.7, 0.3))

    # for model, name in zip(models, model_names):
    model.fit(train.X, train.y)

    y_pred = model.predict(test.X)
    y_pred = y_pred.ravel()
    
    # classify
    for i, y_p in enumerate(y_pred):
        if y_p > 0.5: # TODO: should be 0?
            y_pred[i] = 1
        else:
            y_pred[i] = -1
    wrong = util.percent_wrong(y_pred.ravel(), test.y.ravel())
    acc = 1.0 - wrong
    print(model.name, ' : acc:', acc)

if __name__ == "__main__":
    # parser = create_parser()
    # args = parser.parse_args()
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
    svm = SVC()
    svm.name = 'SVM'
    models = [
        # rff(D=10),
        # rff(D=100),
        # rff(D=1000),
        # Nystrom_Ridge(10),
        # Nystrom_Ridge(100),
        # Nystrom_Ridge(1000),
        # svm,
        ManifoldRLS('rbf', 0.1),
        LapRLS('rbf', 0.1, 0.1),
        kernel.RidgeKernel('rbf', 1.0),
    ]
    tests = [
        lambda model : sphere_test(model, d=1000),
        lambda model : checkerboard_test(model, seed=1, noise=0.0),
        lambda model : checkerboard_test(model, seed=1, noise=0.2),
        lambda model : adult_test(model)
    ]
    test_names = [
        'sphere', 'checkerboard', 'checkerboard_noise', 'adult'
    ]

    # X, y = checkerboard.generate_data((100000, 2), noise=0.1, seed=1, shuffle=False)
    # labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y]
    # plt.scatter(X[:,0], X[:,1], c=labels)
    # plt.show()

    test_runner(models, tests, test_names)

    
    
    
# def create_parser():
#     parser = argparse.ArgumentParser(description='Kernel Method Bake-Off')

#     parser.add_argument('--model', '-m', type=str, default='rff')
#     parser.add_argument('--dataset', '-d', type=str, default='adult')
#     parser.add_argument('--kernel', '-k', type=str, default='gaussian')
#     parser.add_argument('--Dimensions', '-D', type=int, default=100)
#     parser.add_argument('--n_samples', '-n', type=int, default=1000)
#     parser.add_argument('--C', '-c', type=float, default=1.0)

#     return parser

# def execute(args):
#     m = args.model
#     if m == 'rff':
#         model = rff(D = args.Dimensions, k = args.kernel)
#     if m == 'SVC':
#         model = SVC()
#     if m == 'SVM':
#         model = SVM(kernel = args.kernel, c = args.C)
    
#     if args.dataset == 'adult':
#         adult_test([model], [args.model])
#     elif args.dataset == 'sphere':
        # sphere_test([model], [args.model], show_plots=True)
    
