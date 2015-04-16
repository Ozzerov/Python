__author__ = 'alex'

import pandas as pd
import numpy as np
from itertools import product
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer



def logloss(p, b):
    p = np.clip(p/p.sum(axis=1, keepdims=True), 1e-15, 1 - 1e-15)
    return -np.sum(b*np.log(p))/p.shape[0]



df = pd.DataFrame.from_csv('train.csv')
df.target = df.target.apply(lambda x: int(x[-1])) - 1
cut = round(.9*len(df.index))


p = np.ones((1, 9))/9

b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

print(logloss(p, b))

"""
sdf = df.reindex(np.random.permutation(df.index)).reset_index()
Train_y = sdf.target.values[:cut]
Train_X = np.log(sdf.drop(['target', 'id'], axis=1).values[:cut] + 1)
Test_y = sdf.target.values[cut:]
Test_X = np.log(sdf.drop(['target', 'id'], axis=1).values[cut:] + 1)

lb = LabelBinarizer().fit(list(range(9)))
Test_b = lb.transform(Test_y)
print(Test_y)
print(Test_b)

clf = SGDClassifier(loss='log', n_iter=2000).fit(Train_X, Train_y)
P = clf.predict_proba(Test_X)
print(P)

print(logloss(P, Test_b))
"""

