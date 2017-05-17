import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
import LR_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn import metrics


def fold_valid(clf, X, y):
    n_folds = 10
    np.random.seed(0)
    shuffle = True
    if shuffle:
        idx = np.random.permutation(y.shape[0])
        X = X[idx]
        y = y[idx]
    skf = list(StratifiedKFold(y, n_folds))
    auc_set = []
    for i, (train, test) in enumerate(skf):
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        roc_auc = metrics.roc_auc_score(y_test, preds)
        auc_set.append(roc_auc)
    average_auc = np.average(auc_set)
    return average_auc, auc_set


def validation(clf, X, y):
    seed = 1220
    train, val, train_y, val_y = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    clf.fit(train, train_y)
    preds = clf.predict_proba(val)[:, 1]
    roc_auc = metrics.roc_auc_score(val_y, preds)
    # print('val_auc:', roc_auc)
    return roc_auc


if __name__ == '__main__':
    X = LR_data.load_train_oneHot()
    test_x = LR_data.load_test_oneHot()
    test_uid = test_x[:, 0]
    test_x = np.delete(test_x, 0, 1)
    X = np.delete(X, 0, 1)  # delete uid
    y = X[:, -1]
    y = np.asarray(y)
    X = np.delete(X, -1, 1)  # delete y
    X_r, X_c = X.shape  # 15000 1758
    feaScore = LR_data.load_feaScore()   # 1321

    num = 1
    i = 0
    cur = []
    last = 0
    resultFile = open('data/result.csv', 'a')
    while i < len(feaScore):
        cur.append(feaScore[i])
        i += 1
        cur_X = X[:, cur]
        lr = LogisticRegression(class_weight='balanced')
        val, auc_set = fold_valid(lr, cur_X, y)
        if val >= last:
            print(val, auc_set)
            print(cur)
            resultFile.write('%s\n%f%s\n' % (cur, val, auc_set))
            last = val
        else:
            cur.pop()
    resultFile.close()

    # feaScore1 = feaScore[0:55]
    # feaScore1.extend(feaScore[85:100])
    # feaScore1.extend(feaScore[110:115])
    # feaScore1.extend(feaScore[130:135])
    # feaScore1.extend(feaScore[165:170])
    # feaScore1.extend(feaScore[190:195])
    # feaScore1.extend(feaScore[200:205])
    # feaScore1.extend(feaScore[215:220])
    # feaScore1.extend(feaScore[225:230])
    # feaScore1.extend(feaScore[250:255])

    # feaScore = feaScore1
