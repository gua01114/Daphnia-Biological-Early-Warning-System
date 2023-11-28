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
from sklearn.ensemble import RandomForestClassifier
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
rf_run = RandomForestClassifier(random_state=0, n_jobs=-1)
rf_run.get_params()

def bo_params_rf(max_depth, min_samples_leaf, min_samples_split, max_samples,n_estimators):
    params = {'max_depth': int(max_depth),
              'min_samples_leaf': int(min_samples_leaf),
              'min_samples_split': int(min_samples_split),
              'max_samples': max_samples,
              'n_estimators': int(n_estimators)}
    clf = RandomForestClassifier(random_state=0,
                                 max_depth=params['max_depth'],
                                 min_samples_leaf=params['min_samples_leaf'],
                                 min_samples_split=params['min_samples_split'],
                                 max_samples=params['max_samples'],
                                 n_estimators=params['n_estimators'])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_train)
    score = metrics.accuracy_score(preds,y_train)
    return score

rf_BO = BayesianOptimization(bo_params_rf, {'max_depth': (60,100),
                                            'min_samples_leaf': (1,4),
                                            'min_samples_split': (2,10),
                                            'max_samples': (0.5,1),
                                            'n_estimators': (100, 250)})
results = rf_BO.maximize(n_iter=100, init_points=20,acq='ei')
params = rf_BO.max['params']
params['max_depth']= int(params['max_depth'])
params['min_samples_leaf']= int(params['min_samples_leaf'])
params['min_samples_split']= int(params['min_samples_split'])
params['n_estimators']= int(params['n_estimators'])
print('Optimum hyper parameter:', params)

# 05. Model Training
rf_run = RandomForestClassifier(random_state=0,
                                max_depth=params['max_depth'],
                                min_samples_leaf=params['min_samples_leaf'],
                                min_samples_split=params['min_samples_split'],
                                max_samples=params['max_samples'],
                                n_estimators=params['n_estimators'])
rf_run.fit(X_train, y_train)
    # train
train_predict = rf_run.predict(X_train)
print("The training accuracy of the Random Forests model is :\t", metrics.accuracy_score(train_predict,y_train))
    # validation
test_predict = rf_run.predict(X_test)
print("The test accuracy of the Random Forests model is :\t", metrics.accuracy_score(test_predict,y_test))