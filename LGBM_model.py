## 00. Clear
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]

import warnings
warnings.filterwarnings(action='ignore')

## 01. Import library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from sklearn import metrics
from sklearn.metrics import classification_report

## 02. Data loading
data = pd.read_excel()

## 03. Data preprocessing
X = data.drop([],axis=1)
Y = data.loc[:,]
dim = df.nunique(Y)
# Split the data
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

## 04. Hyper parameter tuning
lgb_run = lgb.LGBMClassifier(random_state=0, n_jobs=-1)
lgb_run.get_params()

def bo_params_lgb(max_depth, learning_rate, n_estimators, num_leaves, subsample, subsample_freq, colsample_bytree, reg_lambda, reg_alpha):
    params = {'max_depth': int(max_depth),
              'learning_rate': learning_rate,
              'n_estimators': int(n_estimators),
              'num_leaves': int(num_leaves),
              'subsample': subsample,
              'subsample_freq': int(subsample_freq),
              'colsample_bytree': colsample_bytree,
              'reg_lambda': reg_lambda,
              'reg_alpha': reg_alpha}
    clf = lgb.LGBMClassifier(random_state=0,
                             max_depth=params['max_depth'],
                             learning_rate=params['learning_rate'],
                             num_leaves=params['num_leaves'],
                             subsample=params['subsample'],
                             subsample_freq=params['subsample_freq'],
                             colsample_bytree=params['colsample_bytree'],
                             reg_lambda=params['reg_lambda'],
                             reg_alpha=params['reg_alpha'])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_train)
    score = metrics.accuracy_score(preds, y_train)
    return score

lgb_BO = BayesianOptimization(bo_params_lgb, {'max_depth': (-1,256),
                                              'learning_rate': (0.01, 1.0),
                                              'n_estimators': (30, 5000),
                                              'num_leaves': (2, 512),
                                              'subsample': (0.01, 1.0),
                                              'subsample_freq': (1, 10),
                                              'colsample_bytree': (0.01, 1.0),
                                              'reg_lambda': (1e-9, 100.0),
                                              'reg_alpha': (1e-9, 100.0)},
                              random_state=1)
results = lgb_BO.maximize(n_iter=200, init_points=20,acq='ei')
params = lgb_BO.max['params']
params['max_depth']= int(params['max_depth'])
params['n_estimators']= int(params['n_estimators'])
params['num_leaves']= int(params['num_leaves'])
params['subsample_freq']= int(params['subsample_freq'])
print('Optimum hyper parameter:', params)

# 05. Model Training
lgb_run = lgb.LGBMClassifier(random_state=0,
                             max_depth=params['max_depth'],
                             learning_rate=params['learning_rate'],
                             num_leaves=params['num_leaves'],
                             subsample=params['subsample'],
                             subsample_freq=params['subsample_freq'],
                             colsample_bytree=params['colsample_bytree'],
                             reg_lambda=params['reg_lambda'],
                             reg_alpha=params['reg_alpha']
)
lgb_run.fit(X_train, y_train)
    # train
train_predict = lgb_run.predict(X_train)
print("The training accuracy of the LGBM model is :\t", metrics.accuracy_score(train_predict,y_train))
    # validation
test_predict = lgb_run.predict(X_test)
print("The test accuracy of the LGBM model is :\t", metrics.accuracy_score(test_predict,y_test))
