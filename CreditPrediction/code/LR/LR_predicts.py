import numpy as np
import LR_data
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics


def lr_predict():
    pass


if __name__ == '__main__':
    select_fea = [292, 779, 747, 75, 803, 366, 956, 137, 142, 58, 192, 790, 963, 619, 604, 445, 160, 596, 952, 733, 701, 94, 208, 1007, 953, 434, 226, 143, 101, 164, 78, 135, 151, 34, 119, 572, 223, 97, 59, 955, 120, 570, 841, 130, 835, 697, 657, 345, 213, 854, 577, 194, 157, 621, 21, 784, 149, 199, 138,
                  122, 837, 108, 665, 33, 6, 171, 394, 189, 209, 622, 537, 506, 495, 140, 73, 144, 728, 358, 646, 129, 50, 382, 502, 17, 304, 159, 174, 576, 355, 53, 185, 110, 356, 230, 1036, 45, 542, 880, 562, 742, 291, 148, 776, 294, 102, 430, 958, 446, 127, 568, 214, 428, 153, 1067, 400, 22, 921, 1041, 991, 493]
    # select_fea = [c-1 for c in select_fea1]
    seed = 1220
    # X = LR_data.load_train_origin()
    # test_x = LR_data.load_test_origin()
    X = LR_data.load_train_oneHot()
    test_x = LR_data.load_test_oneHot()
    test_uid = test_x[:, 0]
    test_x = np.delete(test_x, 0, 1)
    X = np.delete(X, 0, 1)  # delete uid
    y = X[:, -1]
    y = np.asarray(y)
    X = np.delete(X, -1, 1)  # delete y
    # feaScore = LR_data.load_feaScore()
    # select_fea = feaScore
    X = X[:, select_fea]
    test_x = test_x[:, select_fea]
    lr = LogisticRegression(class_weight='balanced', C=0.0001)
    # lr.fit(X, y)
    train, val, train_y, val_y = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    lr.fit(train, train_y)
    val_pre = lr.predict_proba(val)[:, 1]
    roc_auc = metrics.roc_auc_score(val_y, val_pre)
    print('roc_auc:', roc_auc)
    result = lr.predict_proba(test_x)[:, 1]
    with open('data/lr_result.csv', 'w', encoding='utf-8') as f:
        f.write('"uid","score"\n')
        for i, v in zip(test_uid, result):
            f.write('%d,%f\n' % (int(i), v))
