__author__ = 'alex'
import pandas as pd
import numpy as np
import xgboost as xgb


bst2 = xgb.Booster(model_file='s4487d8w4.model')


df = pd.DataFrame.from_csv('test.csv')

X = np.log(df.values + 1)

print(X.shape)

test = xgb.DMatrix(X)

preds2 = bst2.predict(test)

print(preds2)



df = pd.DataFrame(preds2, columns=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5',
                               'Class_6', 'Class_7', 'Class_8', 'Class_9'])

df.index.name = 'id'
df.index += 1
df.to_csv('x.csv')
print(df)