import csv
import numpy as np


def reformat():
    def _load(filename):
        with open(filename) as f:
            next(f)
            X = [[float(c) for c in row[0:]] for row in csv.reader(f)]  # 除去ID
        return np.asarray(X)

    with open('data/train_y.csv') as f:
        next(f)
        y = [row[-1] for row in csv.reader(f)]
    np.savez('cache/train.npz', X=_load('data/train_x.csv'), y=np.asarray(y))
    np.savez('cache/test.npz', X=_load('data/test_x.csv'))


def reformat_oneHot():
    def _load(filename):
        with open(filename) as f:
            next(f)
            X = [[float(c) for c in row[0:]] for row in csv.reader(f)]  # 除去ID
        return np.asarray(X)
    np.savez('cache/train_oneHot.npz',
             X=_load('data/01/train_x_oneHot.csv'))  # last column is y
    np.savez('cache/test_oneHot.npz', X=_load('data/01/test_x_oneHot.csv'))


def load_train():
    with open('data/features_type.csv') as f:
        next(f)
        ftype = [row[1] for row in csv.reader(f)]
    train = np.load('cache/train.npz')
    return train['X'], train['y'], ftype


def load_test():
    test = np.load('cache/test.npz')
    return test['X']
    

def load_train_oneHot():
    train = np.load('F:/CreditPrediction/code/cache/train_oneHot.npz')
    return train['X']


def load_test_oneHot():
    test = np.load('F:/CreditPrediction/code/cache/test_oneHot.npz')
    return test['X']


def load_feaInfo():
    with open('data/featureInfo/catefea.csv') as f:
        feaInfo = [[float(c) for c in fea[0:]] for fea in csv.reader(f)]
        feaInfo = np.asarray(feaInfo)
    return feaInfo


def load_feaMissInfo():
    with open('data/feaMissInfo.csv') as f:
        feaMissInfo = [[float(c) for c in fea[0:]] for fea in csv.reader(f)]
        feaMissInfo = np.asarray(feaMissInfo)
    return feaMissInfo


def load_reduc_fea():
    with open('data/redundant_features.txt') as f:
        reduc_f = [int(row[0]) - 1 for row in csv.reader(f)]
    return reduc_f


def load_feaScore():
    with open('F:/CreditPrediction/code/feature_score/feature_score.csv') as f:
        feaScore = [int(row[0]) - 1 for row in csv.reader(f)]
    return feaScore


def submit(pred):
    pass


if __name__ == '__main__':
    reformat_oneHot()
    print('Complete')
