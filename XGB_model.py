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
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
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
xgb_run = XGBClassifier(random_state=0, n_jobs=-1)
xgb_run.get_params()

def bo_params_xgb(max_depth, colsample_bytree, colsample_bylevel, colsample_bynode, gamma, learning_rate, n_estimators, subsample, min_child_weight):
    params = {'max_depth': int(max_depth),
              'colsample_bytree': colsample_bytree,
              'colsample_bylevel': colsample_bylevel,
              'colsample_bynode': colsample_bynode,
              'gamma': int(gamma),
              'learning_rate': learning_rate,
              'n_estimators': int(n_estimators),
              'subsample':subsample,
              'min_child_weight':min_child_weight}
    clf = XGBClassifier(random_state=0,
                        max_depth=params['max_depth'],
                        colsample_bytree=params['colsample_bytree'],
                        colsample_bylevel=params['colsample_bylevel'],
                        colsample_bynode=params['colsample_bynode' ],
                        gamma=params['gamma'],
                        learning_rate=params['learning_rate'],
                        n_estimators=params['n_estimators'],
                        subsample=params['subsample'],
                        min_child_weight=params['min_child_weight'])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_train)
    score = metrics.accuracy_score(preds,y_train)
    return score

xgb_BO = BayesianOptimization(bo_params_xgb, {'max_depth': (1,20),
                                              'colsample_bytree': (0.1,1.0),
                                              'colsample_bylevel': (0.1,1.0),
                                              'colsample_bynode': (0.1,1.0),
                                              'gamma': (0,100),
                                              'learning_rate': (0.01,1.0),
                                              'n_estimators': (100, 2000),
                                              'subsample': (0.5,1),
                                              'min_child_weight': (0,3)},
                              random_state=1)
results = xgb_BO.maximize(n_iter=40, init_points=10,acq='ei')
#results = xgb_BO.maximize(n_iter=100, init_points=10,acq='ei')
params = xgb_BO.max['params']
params['max_depth']= int(params['max_depth'])
params['gamma']= int(params['gamma'])
params['n_estimators']= int(params['n_estimators'])
print('Optimum hyper parameter:', params)

# 05. Model Training
xgb_run = XGBClassifier(random_state=0,
                        max_depth=params['max_depth'],
                        colsample_bytree=params['colsample_bytree'],
                        colsample_bylevel=params['colsample_bylevel'],
                        colsample_bynode=params['colsample_bynode' ],
                        gamma=params['gamma'],
                        learning_rate=params['learning_rate'],
                        n_estimators=params['n_estimators'],
                        subsample=params['subsample'],
                        min_child_weight=params['min_child_weight'])
xgb_run.fit(X_train, y_train)
    # train
train_predict = xgb_run.predict(X_train)
print("The training accuracy of the XGBoost model is :\t", metrics.accuracy_score(train_predict,y_train))
    # validation
test_predict = xgb_run.predict(X_test)
print("The test accuracy of the XGBoost model is :\t", metrics.accuracy_score(test_predict,y_test))