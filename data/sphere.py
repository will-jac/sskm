import numpy as np

import util
# generates two classes for binary classification
# the classes are overlapping circles

rng = np.random.default_rng()

#Get a random point on a unit hypersphere of dimension n
def random_hypersphere_point(d, n, r=1, m=0, sd=1):
    # fill a list of n normal random values
    points = rng.normal(m, sd, size=(n, d))
    # calculate 1 / sqrt of sum of squares
    sqr_red = 1.0 / np.sqrt(np.sum(np.square(points), axis=1))

    # multiply each point by scale factor 1 / sqrt of sum of squares
    # return map(lambda x: x * sqr_red, points)

    return [[points[i,j] * sqr_red[i] *r for j in range(d)] for i in range(n)]

def generate_data(d=2, n=1000, r_1=1, r_2=2, u=None):
    x1 = random_hypersphere_point(d, n, r_1)
    x2 = random_hypersphere_point(d, n, r_2)
    X = x1 + x2
    X = np.array(X)
    y = np.array([0 for _ in range(n)] + [1 for _ in range(n)])

    # shuffle the data
    X, y = util.shuffle(X, y)

    # y.shape = (-1, 1)
    if u is not None:
        # print('generating unlabeled sphere data')
        u1 = random_hypersphere_point(d, u, r_1)
        u2 = random_hypersphere_point(d, u, r_2)
        # print('U has components:', len(u1), len(u2)) 
        U = u1 + u2
        U = np.array(U)
        np.random.shuffle(U)
        # print('U is:', U.shape)
        return X, y, U
    else:
        return X, y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y = generate_data(n=1000)
    labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y]
    print(y.shape, len(labels))
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.show()