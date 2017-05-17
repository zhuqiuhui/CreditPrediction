# coding=utf-8
from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
import sys
sys.path.append('D:\\CreditPredict\\xgboost-master\\xgboost-master\\wrapper')


random_seed = 1220

train_xy = pd.read_csv("D:\\CreditPredict\\zqh\\data\\train_x_oneHot.csv")
test = pd.read_csv("D:\\CreditPredict\\zqh\\data\\test_x_oneHot.csv")
test_uid = test.uid
test_x = test.drop(['uid'], axis=1)


# split train set,generate train,val,test set
train_xy = train_xy.drop(['uid'], axis=1)
train, val = train_test_split(
    train_xy, test_size=0.2, random_state=random_seed)
y = train.y  # 'y'为类标的索引
X = train.drop(['y'], axis=1)
val_y = val.y
val_X = val.drop(['y'], axis=1)

# xgboost start here
dtest = xgb.DMatrix(test_x, missing=-1)
dtrain = xgb.DMatrix(X, label=y, missing=-1)
print("execute here!")
dval = xgb.DMatrix(val_X, label=val_y, missing=-1)
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
    'nthread': 4
}


watchlist = [(dval, 'val'), (dtrain, 'train')]
model = xgb.train(params, dtrain, num_boost_round=8000, evals=watchlist)
model.save_model('xgb.model')

test_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid", "score"])
test_result.uid = test_uid
test_result.score = test_y
test_result.to_csv("result.csv", index=None, encoding='utf-8')
