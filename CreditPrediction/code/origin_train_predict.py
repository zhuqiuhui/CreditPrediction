# coding=utf-8
import sys
import os
sys.path.append('E:\\xgboost-master\\xgboost-master\\wrapper')

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb

random_seed = 1225

train_x_csv = "F:\\loan competetion\\train_x.csv"
train_y_csv = "F:\\loan competetion\\train_y.csv"
test_x_csv = "F:\\loan competetion\\test_x.csv"
features_type_csv = "F:\\loan competetion\\features_type.csv"

train_x = pd.read_csv(train_x_csv)
train_y = pd.read_csv(train_y_csv)
train_xy = pd.merge(train_x, train_y, on='uid')

test = pd.read_csv(test_x_csv)
test_uid = test.uid
test_x = test.drop(['uid'], axis=1)

features_type = pd.read_csv(features_type_csv)
features_type.index = features_type.feature
features_type = features_type.drop('feature', axis=1)
# {'x179': 'numeric', 'x586': 'numeric'},若无['type'],则{'type':{'x269':'numeric',...}}
features_type = features_type.to_dict()['type']

feature_info = {}
features = list(train_x.columns)  # ['uid','x1','x2',...]
features.remove('uid')  # ['x1','x2',...]

for feature in features:
    max_ = train_x[feature].max()
    min = train_x[feature].min()
    n_null = len(train_x[train_x[feature] < 0])  # 为缺失值的个案数
    n_gt1w = len(train_x[train_x[feature] > 10000])  # 大于10000的个案数
    feature_info[feature] = [min, max_, n_null, n_gt1w]

# see how many neg/pos sample
print("neg:{0},pos:{1}".format(
    len(train_xy[train_xy.y == 0]), len(train_xy[train_xy.y == 1])))

# split train set,generate train,val,test set

train_xy = train_xy.drop(['uid'], axis=1)
train, val = train_test_split(
    train_xy, test_size=0.2, random_state=random_seed)
y = train.y  # 'y'为类标的索引
X = train.drop(['y'], axis=1)
val_y = val.y
val_X = val.drop(['y'], axis=1)

# xgboost start here
dtest = xgb.DMatrix(test_x)
dval = xgb.DMatrix(val_X, label=val_y)
dtrain = xgb.DMatrix(X, label=y)
params = {
    'booster': 'gbtree',  # gbtree used
    'objective': 'binary:logistic',
    'early_stopping_rounds': 100,
    'scale_pos_weight': 1600.0 / 13458.0,  # 正样本权重
    'eval_metric': 'auc',
    'gamma': 0.1,
    'max_depth': 8,
    'lambda': 700,
    'subsample': 0.7,
    'colsample_bytree': 0.3,
    'min_child_weight': 5,
    'eta': 0.02,
    'seed': random_seed,
    'nthread': 7
}

watchlist = [(dval, 'val'), (dtrain, 'train')]
model = xgb.train(params, dtrain, num_boost_round=40000, evals=watchlist)
model.save_model('C:/xgb.model')

# predict test set (from the best iteration)
test_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid", "score"])
test_result.uid = test_uid
test_result.score = test_y
# remember to edit xgb.csv , add “”
test_result.to_csv("C:/xgb.csv", index=None, encoding='utf-8')

# save feature score and feature information:
# feature,score,min,max,n_null,n_gt1w
feature_score = model.get_fscore()
for key in feature_score:
    feature_score[key] = [feature_score[key]] + \
        feature_info[key] + [features_type[key]]

feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
fs = []
for (key, value) in feature_score:
    fs.append("{0},{1},{2},{3},{4},{5},{6}\n".format(
        key, value[0], value[1], value[2], value[3], value[4], value[5]))

with open('C:/feature_score.csv', 'w') as f:
    f.writelines("feature,score,min,max,n_null,n_gt1w\n")
    f.writelines(fs)
