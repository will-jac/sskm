import numpy as np
import matplotlib.pyplot as plt

import argparse

# ML methods
from random_fourier_features import rff
from nystrom import NystromTransformer

import kernel
from SVM import SVM

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def mse(predict, true, axis=0):
    t = 0
    for i in range(len(predict)):
        t += (predict[i] - true[i])**2
    return np.sqrt(t / len(predict))

# for classification
def percent_wrong(predict, true):
    n = len(predict)
    wrong = 0
    for i in range(n):
        if predict[i] != true[i]:
            wrong += 1
    return 1.0 * wrong / n

from collections import namedtuple

def train_test_valid_split(X, y, split=(0.8, 0.1, 0.1), shuffle=True):
    assert sum(split) == 1
    assert X.shape[0] == y.shape[0]

    # first, shuffle the data
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        # Shuffle the arrays by giving the permutation in the square brackets.
        X, y = X[permutation], y[permutation]
    
    # used to return  data so it can be accessed as train.X, train.y
    Data = namedtuple('Data', ['X', 'y'])

    n = X.shape[0]
    start = 0
    stop = 0
    # Data object used for implicit type checking / inferrence
    split_data = [Data([],[])]*len(split)
    for i in range(len(split)):
        stop  = int(n * sum(split[0:i+1]))
        split_data[i] = Data(X[start:stop,:], y[start:stop])
        start = stop

    return tuple(split_data)

def sphere_test(models, model_names = None, n = 1000, d = 1000, show_plots=True):
    
    if model_names is None:
        model_names = ['model ' + str(i) for i in range(len(models))]

    import data.sphere as sphere
    X, y = sphere.generate_data(n=n, d=d)
    (train, test) = train_test_valid_split(X, y, split=(0.7, 0.3))

    if show_plots:
        f = plt.figure()
        plt.title('Labeled Data (input)')
        labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y]
        plt.scatter(X[:,0], X[:,1], c=labels)
        plt.show()
        plt.close(f)

    for model, name in zip(models, model_names):
        model.fit(train.X, train.y)

        y_pred = model.predict(test.X)
        y_pred = y_pred.ravel()
        
        # classify
        for i, y_p in enumerate(y_pred):
            if y_p > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        wrong = percent_wrong(y_pred.ravel(), test.y.ravel())
        print(name, '%wrong:', wrong)

        if show_plots:
            plt.figure()
            plt.title(f'{name} on sphere (d={d}), %wrong={wrong}')
            # plot (for sphere)
            labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y_pred]
            plt.scatter(test.X[:,0], test.X[:,1], c=labels)
            plt.show()
            plt.close(f)

def checkerboard_test(model, model_name, shape, n_clusters, noise)

def adult_test(models, model_names = None):

    if model_names is None:
        model_names = ['model ' + str(i) for i in range(len(models))]

    import data.adult as adult

    (train, test) = train_test_valid_split(adult.X, adult.y, split=(0.7, 0.3))

    for model, name in zip(models, model_names):
        model.fit(train.X, train.y)

        y_pred = model.predict(test.X)
        y_pred = y_pred.ravel()
        
        # classify
        for i, y_p in enumerate(y_pred):
            if y_p > 0.5: # TODO: should be 0?
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        wrong = percent_wrong(y_pred.ravel(), test.y.ravel())
        print(name, '%wrong:', wrong)

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
    
if __name__ == "__main__":
    # parser = create_parser()
    # args = parser.parse_args()
    from sklearn.kernel_approximation import Nystroem

    class Nystrom_Ridge():
        def __init__(self):
            self.transformer =  NystromTransformer('rbf', n_components=10, gamma=0.2) 
            # self.transformer = Nystroem(gamma=0.2, n_components=10) 
            self.kernel_method = kernel.RidgeKernel('precomputed', 1.0)
        def fit(self, X, y):
            X = self.transformer.fit(X, y).transform(X)
            self.kernel_method.fit(X, y)

        def predict(self, X):
            X = self.transformer.transform(X)
            return self.kernel_method.predict(X)

    models = [
        # rff(D=2),
        rff(D=10),
        # rff(D=100),
        # rff(D=1000),
        SVC(),
        # kernel.RidgeKernel('rbf', 1.0),
        Nystrom_Ridge(), #kernel.RidgeKernel('rbf', 1.0)
        
    ]
    model_names = [
        # 'rff d=2',
        'rff d=10',
        # 'rff d=100',
        # 'rff d=1000',
        'SVC',
        # 'Ridge',
        'Nystroem + Ridge',
        
    ]
    sphere_test(models, model_names, d=1000, show_plots=False)
    adult_test(models, model_names)
    