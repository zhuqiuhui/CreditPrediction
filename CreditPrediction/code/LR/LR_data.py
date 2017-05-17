import csv
import numpy as np


def reformat_oneHot():
    def _load(filename):
        with open(filename) as f:
            next(f)
            X = [[float(c) for c in row[0:]] for row in csv.reader(f)]  # 除去ID
        return np.asarray(X)
    # last column is y
    np.savez('cache/train_label_oneHot.npz', X=_load('data/train_label.csv'))
    np.savez('cache/test_x_oneHot.npz', X=_load('data/test_x.csv'))


def load_train_oneHot():
    train = np.load('cache/train_label_oneHot.npz')
    return train['X']


def load_test_oneHot():
    test = np.load('cache/test_x_oneHot.npz')
    return test['X']


def reformat():
    def _load(filename):
        with open(filename) as f:
            next(f)
            X = [[float(c) for c in row[0:]] for row in csv.reader(f)]  # 除去ID
        return np.asarray(X)
    # last column is y
    np.savez('cache/train_label_origin.npz',
             X=_load('data/train_x_origin.csv'))
    np.savez('cache/test_x_origin.npz', X=_load('data/test_x_origin.csv'))


def load_train_origin():
    train = np.load('cache/train_label_origin.npz')
    return train['X']


def load_test_origin():
    test = np.load('cache/test_x_origin.npz')
    return test['X']


def load_feaScore():
    with open('data/feature_score.csv') as f:
        feaScore = [int(row[0]) for row in csv.reader(f)]
    return feaScore


if __name__ == '__main__':
    # reformat_oneHot()
    reformat()
    print('Complete')
