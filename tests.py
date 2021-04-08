import numpy as np

import util

# data sets
import data.adult as adult
import data.sphere as sphere
import data.checkerboard as checkerboard

# other things
from sklearn.metrics import roc_curve, plot_roc_curve

def test_runner(models, test_funcs, test_funcs_names):
    results = []
    for test_f, test_name in zip(test_funcs, test_funcs_names):
        print('---- running test:', test_name, ' ----')
        test_results = []
        for model in models:
            test_results.append(test_f(model))
        results.append(test_results)
    return results

def sphere_test(model, n=1000, u=1000, d = 1000, show_plots=False):
    if u is None:
        X, y = sphere.generate_data(n=n, d=d)
        (train, test) = util.train_test_valid_split(X, y, split=(0.7, 0.3))
    else:
        print('generating unlabeled data')
        X, y, U = sphere.generate_data(n=n, d=d, u=u)
        # print('U is:', U.shape)
        (train, test) = util.train_test_valid_split(X, y, split=(0.7, 0.3), U=U)
        print('generated data has shape:', train.X.shape, test.X.shape)
    # if show_plots:
    #     import matplotlib.pyplot as plt
    #     f = plt.figure()
    #     plt.title('Labeled Data (input)')
    #     labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y]
    #     plt.scatter(X[:,0], X[:,1], c=labels)
    #     plt.show()
    #     plt.close(f)

    # for model, name in zip(models, model_names):
    if u is None:
        model.fit(train.X, train.y)
    else:
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

    if show_plots:
        import matplotlib.pyplot as plt
        f = plt.figure()
        plt.title(f'{model.name} on sphere (d={d}), %wrong={wrong}')
        # plot (for sphere)
        labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y_pred]
        plt.scatter(test.X[:,0], test.X[:,1], c=labels)
        plt.show()
        plt.close(f)

        # plot_roc_curve(model, test.X, test.y)
        # plt.show()
        # y_pred_binary = [[y, 1-y] for y in y_pred]
        # fpr, tpr, thresh = roc_curve(test.y, y_pred)

        # plt.plot(fpr, tpr)
        # print(thresh)
        # plt.show()

    return acc

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

    return acc

def adult_test(model, u=0.8):

    # (data, _) = util.train_test_valid_split(adult.X, adult.y, split = (0.1, 0.9))
    data = adult

    (label, unlabeled) = util.train_test_valid_split(data.X, data.y, split=(u, 1 - u))

    (train, test) = util.train_test_valid_split(label.X, label.y, split=(0.7, 0.3), U = unlabeled.X)

    # for model, name in zip(models, model_names):
    model.fit(train.X, train.y, train.U)

    y_pred = model.predict(test.X)
    y_pred = y_pred.ravel()

    # classify
    for i, y_p in enumerate(y_pred):
        if y_p > 0.0: # TODO: should be 0?
            y_pred[i] = 1
        else:
            y_pred[i] = -1
    wrong = util.percent_wrong(y_pred.ravel(), test.y.ravel())
    acc = 1.0 - wrong
    print(model.name, ' : acc:', acc)

    return acc

def draw_decision_boundary(model, d=2, lim=(-4,4), n=100, cutoff=0.5):
    import matplotlib.pyplot as plt
    X = np.empty((n**2, 2))
    r = lim[1] - lim[0]
    for i in range(n):
        for j in range(n):
            X[i*n+j] = [r * float(i) / n + lim[0], r * float(j) / n + lim[0]]
    y = model.predict(X)
    labels = ['#1f77b4' if l < cutoff else '#ff7f0e' for l in y]
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.title(model.name)
    plt.show()

tests = {
    'sphere' : sphere_test,
    'checkerboard' : checkerboard_test,
    'adult' : adult_test,
}