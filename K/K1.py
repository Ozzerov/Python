__author__ = 'alex'
import pandas as pd
import xgboost as xgb

data = pd.DataFrame.from_csv('train.csv')
y = data.target.apply(lambda x: x[-1])
data.drop('target', axis=1, inplace=True)
X = data.values
