import numpy as np

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

def train_test_valid_split(X, y, split=(0.8, 0.1, 0.1), shuffle=True, U=None):
    assert sum(split) == 1
    assert X.shape[0] == y.shape[0]

    # first, shuffle the data
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        # Shuffle the arrays by giving the permutation in the square brackets.
        X, y = X[permutation], y[permutation]
        if U is not None:
            np.random.shuffle(U)
    
    # used to return  data so it can be accessed as train.X, train.y
    Data = namedtuple('Data', ['X', 'y', 'U'])

    # train will have all the unlabeled data!
    n = X.shape[0]
    start = 0
    stop = 0
    # Data object used for implicit type checking / inferrence
    split_data = [Data([],[],[])]*len(split)
    # for train, test, valid
    for i in range(len(split)):
        stop  = int(n * sum(split[0:i+1]))
        if i == 0:
            # train will have all the unlabeled data!
            split_data[i] = Data(X[start:stop,:], y[start:stop], U)
        else:
            split_data[i] = Data(X[start:stop,:], y[start:stop], [])
        start = stop
    return tuple(split_data)

def shuffle(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    return a[permutation], b[permutation]