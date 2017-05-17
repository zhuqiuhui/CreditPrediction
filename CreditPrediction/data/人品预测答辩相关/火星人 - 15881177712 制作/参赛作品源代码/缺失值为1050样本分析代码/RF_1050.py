# coding=utf-8
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv


def load_data(filename):
    with open(filename) as f:
            next(f)
            X = [[float(c) for c in row[1:]] for row in csv.reader(f)]  # 除去ID
    return np.asarray(X)


def rf_model(X, y, test):
    RFmodel = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        class_weight='balanced_subsample',
        min_samples_leaf=4)
    RFmodel.fit(X, y)
    test_y = RFmodel.predict_proba(test)[:, 1]
    return test_y


if __name__ == '__main__':
    X = load_data("missTrain_1050.csv")
    y = X[:, -1]
    X = np.delete(X, X.shape[1]-1, 1)
    test_x = load_data("missTest_1050.csv")
    index = []
    for i in range(X.shape[1]):
        value = np.unique(X[:, i], return_counts=True)
        if len(value[0]) > 1:
            index.append(i)
    print(len(index))
    X = X[:, index]
    test_x = test_x[:, index]
    res = rf_model(X, y, test_x)
    print(res)
