# coding=utf-8
from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
import sys
sys.path.append('E:\\xgboost-master\\xgboost-master\\wrapper')


random_seed = 1220

train_xy = pd.read_csv("feature_score/train_x_oneHot_feaScore.csv")
test = pd.read_csv("feature_score/test_x_oneHot_feaScore.csv")
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
dtest = xgb.DMatrix(test_x)

dtrain = xgb.DMatrix(X, label=y)
print("execute here!")
dval = xgb.DMatrix(val_X, label=val_y)
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
    'nthread': 3,
    'max_delta_step': 1
}

"""
early_stopping_rounds (int) – Activates early stopping. Validation error needs to 
decrease at least every <early_stopping_rounds> round(s) to continue training. 
Requires at least one item in evals. If there’s more than one, will use the last.
 Returns the model from the last iteration (not the best one). If early stopping 
 occurs, the model will have three additional fields: bst.best_score, 
 bst.best_iteration and bst.best_ntree_limit. (Use bst.best_ntree_limit to get the
correct value if num_parallel_tree and/or num_class appears in the parameters)
"""
# ntree_limit (int) – Limit number of trees in the prediction; defaults to 0 (use all trees).
# num_boost_round (int) – Number of boosting iterations.
watchlist = [(dval, 'val'), (dtrain, 'train')]
model = xgb.train(params, dtrain, num_boost_round=300, evals=watchlist)
model.save_model('C:/xgb.model')

print('best_ntree_limit:', model.best_ntree_limit)
# predict test set (from the best iteration)
test_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid", "score"])
test_result.uid = test_uid
test_result.score = test_y
test_result.to_csv("C:/result.csv", index=None, encoding='utf-8')
