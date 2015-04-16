__author__ = 'alex'
import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product

#def logloss(p, b):
#    p = np.clip(p/p.sum(axis=1, keepdims=True), 1e-15, 1 - 1e-15)
#    return -np.sum(b*np.log(p))/p.shape[0]

df = pd.DataFrame.from_csv('train.csv')
df.target = df.target.apply(lambda x: int(x[-1])) - 1
cut = round(.9*len(df.index))

for depth, width, i in product([7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10, 11], list(range(2))):

    sdf = df.reindex(np.random.permutation(df.index)).reset_index()
    Train_y = sdf.target.values[:cut]
    Train_X = np.log(sdf.drop(['target', 'id'], axis=1).values[:cut] + 1)
    Test_y = sdf.target.values[cut:]
    Test_X = np.log(sdf.drop(['target', 'id'], axis=1).values[cut:] + 1)

    train = xgb.DMatrix(Train_X, label=Train_y)
    test = xgb.DMatrix(Test_X, label=Test_y)

    param = {}
    param['objective'] = 'multi:softprob'
    param['bst:eta'] = 0.1
    param['bst:max_depth'] = depth
    param['subsample'] = 0.9
    param['eval_metric'] = 'mlogloss'
    param['num_class'] = 9
    param['silent'] = 1
    param['nthread'] = 4
    param['min_child_weight'] = width
    #param['num_feature'] = 15


    watchlist = [(train, 'train'), (test, 'eval')]
    nround = 20000
    bst = xgb.train(param, train, nround, watchlist, early_stopping_rounds=25)
    if bst.best_score < 0.45:
        name = 's' + str(round(bst.best_score*10000)) + 'd' + str(depth) + 'w' + str(width) + '.model'
        bst.save_model(name)


