import numpy as np
import csv


def reformat():
    def _load(filename):
        with open(filename) as f:
            next(f)
            X = [[float(c) for c in row[1:]] for row in csv.reader(f)]  # 除去ID
        return np.asarray(X)

    with open('data/train_y.csv') as f:
        next(f)
        y = [float(row[-1]) for row in csv.reader(f)]
    np.savez('cache/train_x_ens.npz',
             X=_load('data/train_x_ens.csv'), y=np.asarray(y))
    np.savez('cache/test_x_ens.npz', X=_load('data/test_x_ens.csv'))


def load_train():
    train = np.load('cache/train_x_ens.npz')
    return train['X'], train['y']


def load_test():
    test = np.load('cache/test_x_ens.npz')
    return test['X']


if __name__ == '__main__':
    reformat()
    print('over')
