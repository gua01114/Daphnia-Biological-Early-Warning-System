## 00. Clear
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]

import warnings
warnings.filterwarnings(action='ignore')

## 01. Import library
import pandas as pd
import numpy as np
import torch
import pickle

from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import metrics
from bayes_opt import BayesianOptimization
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

## 02. Data loading
data = pd.read_excel()

## 03. Data preprocessing
np.random.seed(1)
if "Set" not in data.columns:
    data["Set"] = np.random.choice(["train", "test"], p =[.7, .3], size=(data.shape[0],))

train_indices = data[data.Set == "train"].index
test_indices = data[data.Set == "test"].index
print(train_indices.shape, test_indices.shape)

## 03. Data preprocessing
target = ''
np.random.seed(1)
if "Set" not in data.columns:
    data["Set"] = np.random.choice(["train", "test"], p =[.7, .3], size=(data.shape[0],))
train_indices = data[data.Set == "train"].index
test_indices = data[data.Set == "test"].index
print(train_indices.shape, test_indices.shape)

nunique = data.nunique()
types = data.dtypes
# Unique values < 200 : Convert to categorical variables
# Unique values > 200 : Fill NA with mean values
categorical_columns = []
categorical_dims =  {}
for col in data.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, data[col].nunique())
        l_enc = LabelEncoder()
        data[col] = data[col].fillna("VV_likely")
        data[col] = l_enc.fit_transform(data[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        data.fillna(data.loc[train_indices, col].mean(), inplace=True)
print(data.head())

# Saving the index and dimension of categorical variables for categorical Embedding
unused_feat = ['Set']
features = [ col for col in data.columns if col not in unused_feat+[target]]
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# Split data train, valid and test
X_train = data[features].values[train_indices]
y_train = data[target].values[train_indices]

X_test = data[features].values[test_indices]
y_test = data[target].values[test_indices]

## 04. Hyper parameter tuning
TN_run = TabNetClassifier()
TN_run.get_params()

def bo_params_TN(n_d, n_a, n_steps, gamma, n_independent, n_shared, momentum):
    params = {'n_d': int(n_d),
              'n_a': int(n_a),
              'n_steps': int(n_steps),
              'gamma': float(gamma),
              'n_independent': int(n_independent),
              'n_shared': int(n_shared),
              'momentum': float(momentum),
              }

    clf = TabNetClassifier(n_d = params['n_d'],
                           n_a = params['n_a'],
                           n_steps = params['n_steps'],
                           gamma = params['gamma'],
                           cat_idxs=cat_idxs,
                           cat_dims=cat_dims,
                           cat_emb_dim=1,
                           n_independent = params['n_independent'],
                           n_shared = params['n_shared'],
                           momentum =params['momentum'],
                           optimizer_params=dict(lr=2e-2),
                           optimizer_fn=torch.optim.Adam,
                           scheduler_fn= torch.optim.lr_scheduler.StepLR,
                           scheduler_params={"gamma": 0.95,
                                             "step_size": 20},
                           epsilon=1e-15)
    clf.fit(X_train, y_train, max_epochs=30)
    preds = clf.predict(X_train)
    score = metrics.accuracy_score(preds, y_train)
    return score

TN_BO = BayesianOptimization(bo_params_TN, {'n_d': (8, 64),
                                            'n_a': (8, 64),
                                            'n_steps': (3, 10),
                                            'gamma': (1.0, 2.0),
                                            'n_independent': (1, 5),
                                            'n_shared': (1, 5),
                                            'momentum': (0.01, 0.4)},
                             random_state=1)
results = TN_BO.maximize(n_iter=100, init_points=10,acq='ei')
params = TN_BO.max['params']
params['n_d']= int(params['n_d'])
params['n_a']= int(params['n_a'])
params['n_steps']= int(params['n_steps'])
params['n_independent']= int(params['n_independent'])
params['n_shared']= int(params['n_shared'])
print('Optimum hyper parameter:', params)

# 05. Model Training
TN_run = TabNetClassifier(n_d = params['n_d'],
                          n_a = params['n_a'],
                          n_steps = params['n_steps'],
                          gamma = params['gamma'],
                          cat_idxs=cat_idxs,
                          cat_dims=cat_dims,
                          cat_emb_dim=1,
                          n_independent = params['n_independent'],
                          n_shared = params['n_shared'],
                          momentum =params['momentum'],
                          optimizer_params=dict(lr=2e-2),
                          optimizer_fn=torch.optim.Adam,
                          scheduler_fn= torch.optim.lr_scheduler.StepLR,
                          scheduler_params={"gamma": 0.95,
                                            "step_size": 20},
                          epsilon=1e-15)
TN_run.fit(X_train, y_train, max_epochs=200, patience=30)
    # train
train_predict = TN_run.predict(X_train)
print("The training accuracy of the TabNet model is :\t", metrics.accuracy_score(train_predict,y_train))
    # validation
test_predict = TN_run.predict(X_test)
print("The test accuracy of the TabNet model is :\t", metrics.accuracy_score(test_predict,y_test))