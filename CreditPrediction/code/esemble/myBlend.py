import numpy as np
import pandas as pd
import load_my_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import sys
sys.path.append('E:\\xgboost-master\\xgboost-master\\wrapper')


def model_esemble():
    np.random.seed(1220)
    random_seed = 1220
    n_folds = 2
    shuffle = False

    X, y = load_my_data.load_train()
    X_row, X_cloumn = X.shape
    X_submission = load_my_data.load_test()
    X_submission_r, _ = X_submission.shape
    if shuffle:
        idx = np.random.permutation(X_row)
        X = X[idx]
        y = y[idx]
        print(idx)
    skf = list(StratifiedKFold(y, n_folds))
    # print(skf)

    params = {
        'booster': 'gbtree',  # gbtree used
        'objective': 'binary:logistic',
        'early_stopping_rounds': 100,
        'scale_pos_weight': 0.13,  # 正样本权重
        'eval_metric': 'auc',
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 550,
        'subsample': 0.7,
        'colsample_bytree': 0.4,
        'min_child_weight': 3,
        'eta': 0.02,
        'seed': random_seed,
        'nthread': 5
    }

    clfs = ['xgboost',
            RandomForestClassifier(n_estimators=1500,
                                   criterion='gini',
                                   max_depth=8,
                                   class_weight='balanced_subsample',
                                   min_samples_leaf=3),
            RandomForestClassifier(n_estimators=5,
                                   criterion='entropy',
                                   max_depth=8,
                                   class_weight='balanced_subsample',
                                   min_samples_leaf=3),
            ExtraTreesClassifier(n_estimators=1500,
                                 criterion='gini',
                                 max_depth=8,
                                 class_weight='balanced_subsample',
                                 min_samples_leaf=3),
            ExtraTreesClassifier(n_estimators=5,
                                 criterion='entropy',
                                 max_depth=8,
                                 class_weight='balanced_subsample',
                                 min_samples_leaf=3)
            ]
    dataset_blend_train = np.zeros((X.shape[0],
                                    len(clfs)))  # X's row number * 5
    dataset_blend_test = np.zeros((X_submission_r,
                                   len(clfs)))  # test's row number * 5
    notXgb_X_submission = X_submission
    flag = 0
    for j, clf in enumerate(clfs):
        # print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission_r, len(skf)))
        for i, (train, test) in enumerate(skf):
            print('flag:', flag)
            flag = flag + 1
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            if clf == 'xgboost':
                X_train = pd.DataFrame(X_train)
                y_train = pd.DataFrame(y_train)
                X_test = pd.DataFrame(X_test)
                X_test = xgb.DMatrix(X_test)
                if i == 0:
                    X_submission = pd.DataFrame(X_submission)
                    X_submission = xgb.DMatrix(X_submission)
                X_train = xgb.DMatrix(X_train, label=y_train)
                watchlist = [(X_train, 'train')]
                xgb_model = xgb.train(
                    params, X_train, num_boost_round=8, evals=watchlist)
                y_submission = xgb_model.predict(
                    X_test, ntree_limit=xgb_model.best_ntree_limit)
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = xgb_model.predict(X_submission)
            else:
                clf.fit(X_train, y_train)
                y_submission = clf.predict_proba(X_test)[:, 1]
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(
                    notXgb_X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / \
        (y_submission.max() - y_submission.min())

    print("Saving Results.")
    np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')


if __name__ == '__main__':
    model_esemble()
