import numpy as np
import load_model_data
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import sys
sys.path.append('E:\\xgboost-master\\xgboost-master\\wrapper')


def model_esemble():
    np.random.seed(1220)
    random_seed = 1220
    shuffle = True
    train_xy = load_model_data.load_train()
    X_row = len(train_xy.index)
    test_x, test_uid = load_model_data.load_test()
    test_row = len(test_x.index)
    if shuffle:
        idx = np.random.permutation(X_row)
        train_xy = train_xy.ix[idx]

    train, val = train_test_split(
        train_xy, test_size=0.2, random_state=random_seed)
    y = train.y
    X = train.drop(['y'], axis=1)
    val_y = val.y
    val_X = val.drop(['y'], axis=1)

    gdbt_params = {
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
        'nthread': 4
    }
    print('execute here')

    clfs = ['xgboost',
            RandomForestClassifier(n_estimators=3,
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
    dataset_blend_train = np.zeros(
        (len(val_X.index), len(clfs)))  # X's row number * 5
    dataset_blend_test = np.zeros(
        (test_row, len(clfs)))  # test's row number * 5
    flag = 0
    xgb_val_x = xgb.DMatrix(val_X)
    xgb_test_x = xgb.DMatrix(test_x)
    for j, clf in enumerate(clfs):
        print('flag:', flag)
        if clf == 'xgboost':
            X_train = xgb.DMatrix(X, label=y)
            watchlist = [(X_train, 'train')]
            xgb_model = xgb.train(
                gdbt_params, X_train, num_boost_round=8, evals=watchlist)
            val_y_pre = xgb_model.predict(
                xgb_val_x, ntree_limit=xgb_model.best_ntree_limit)
            dataset_blend_train[:, j] = val_y_pre
            dataset_blend_test[:, j] = xgb_model.predict(
                xgb_test_x, ntree_limit=xgb_model.best_ntree_limit)
        else:
            clf.fit(X, y)
            val_y_pre = clf.predict_proba(val_X)[:, 1]
            dataset_blend_train[:, j] = val_y_pre
            dataset_blend_test[:, j] = clf.predict_proba(test_x)[:, 1]
        flag = flag + 1

    print("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, val_y)
    test_y_pre = clf.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    test_y_pre = (test_y_pre - test_y_pre.min()) / \
        (test_y_pre.max() - test_y_pre.min())

    print("Saving Results.")
    np.savetxt(fname='test.csv', X=test_y_pre, fmt='%0.9f')


if __name__ == '__main__':
    model_esemble()
