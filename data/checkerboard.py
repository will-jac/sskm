import numpy as np

import util

def generate_data(shape=(1000000, 2), board_range=2, noise=10, seed=None, shuffle = True):
    board_range = float(board_range)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-board_range, board_range, shape)
    y = np.zeros(shape[0])
    noise_array = rng.normal(0, noise, shape)
    
    def gen_target(z):
        x = z[0] + z[2]
        y = z[1] + z[3]
        # divide into 4x4 clusters
        # this is the absolute worst way to do this
        if x / 2 > 0.5:
            if y / 2 > 0.5:
                return 1
            if y > 0:
                return 0
            elif y / 2 < -0.5:
                return 0
            return 1
        elif x > 0:
            if y / 2 > 0.5:
                return 0
            elif y > 0:
                return 1
            elif y / 2 < -0.5:
                return 1
            return 0
        elif x / 2 > -0.5:
            if y / 2 > 0.5:
                return 1
            elif y > 0:
                return 0
            elif y / 2 < -0.5:
                return 0
            return 1
        else:
            if y / 2 > 0.5:
                return 0
            elif y > 0:
                return 1
            elif y / 2 < -0.5:
                return 1
            return 0
    y = np.apply_along_axis(gen_target, 1, np.c_[X, noise_array])

    if shuffle:
        return util.shuffle(X, y)
    return X, y

if __name__ == '__main__':
    generate_data((10000, 2), noise=0)