## 00. Clear
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]

import warnings
warnings.filterwarnings(action='ignore')

## 01. Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, BatchNormalization, MaxPooling1D, Flatten

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
## Step 1: Initialize space or a required range of values
units_options = np.arange(32, 1024 + 1, 32, dtype=int)
dropout_options = np.arange(.20,.50 + 0.01, 0.025, dtype=float)
batchsize_options = np.arange(32, 128 + 1, 32, dtype=int)
optimizer_option = ['Adadelta', 'Adam', 'RMSprop']
kernelsize_option = np.arange(1, 3 + 1, 1, dtype=int)
poolsize_option = np.arange(1, 3 + 1, 1, dtype=int)
filters_options = np.arange(8, 128 + 1, 8, dtype=int)

space = {'choice': hp.choice('num_layers',
                             [ {'layers':'two', },
                               {'layers':'three',
                                'filter3': hp.choice('filter3', filters_options),
                                'dropout3': hp.choice('dropout3', dropout_options),
                                'kernel3': hp.choice('kernel3', kernelsize_option)
                                }
                               ]
                             ),
         'units1': hp.choice('units1', units_options),
         'units2': hp.choice('units2', units_options),

         'filter1': hp.choice('filter1', filters_options),
         'filter2': hp.choice('filter2', filters_options),

         'kernel1': hp.choice('kernel1', kernelsize_option),
         'kernel2': hp.choice('kernel2', kernelsize_option),

         'dropout1': hp.choice('dropout1', dropout_options),
         'dropout2': hp.choice('dropout2', dropout_options),

         'pool1': hp.choice('pool1', poolsize_option),

         'batch_size' : hp.choice('batch_size', batchsize_options),

         'nb_epochs' :  10,
         'optimizer': hp.choice('optimizer',['Adadelta', 'Adam', 'RMSprop']),
         'activation': 'relu'
         }

## Step 2: Define objective function
def f_nn(params):
    model = Sequential()
    model.add(Conv1D(filters=params['filter1'], kernel_size=int(params['kernel1']), input_shape=train_x.shape[1:]))
    #model.add(BatchNormalization())
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))
    model.add(MaxPooling1D(pool_size=int(params['pool1'])))
    model.add(Conv1D(filters=params['filter2'], kernel_size=int(params['kernel2'])))
    #model.add(BatchNormalization())
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))
    if params['choice']['layers']== 'three':
        model.add(Conv1D(filters=params['choice']['filter3'], kernel_size=int(params['choice']['kernel3'])))
        #model.add(BatchNormalization())
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))
    model.add(Flatten())
    model.add(Dense(units=params['units1']))
    model.add(Activation('relu'))
    model.add(Dense(units=params['units2']))
    model.add(Activation('relu'))

    model.add(Dense(dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=params['nb_epochs'], batch_size=params['batch_size'])
    scores = model.evaluate(train_x, train_y)
    for i, m in enumerate(model.metrics_names):
        print("\n%s: %.3f" % (m, scores[i]))
    return {'loss': scores[0], 'status': STATUS_OK, 'model': model}

## Step 3: Run Hyperopt function
trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=20, trials=trials)
print('\nBest params found:\n', best)

# 05. Model building
batch_size = batchsize_options[best['batch_size']]
dropout1 = dropout_options[best['dropout1']]
dropout2 = dropout_options[best['dropout2']]
filter1 = filters_options[best['filter1']]
filter2 = filters_options[best['filter2']]
kernel1 = kernelsize_option[best['kernel1']]
kernel2 = kernelsize_option[best['kernel2']]
units1 = units_options[best['units1']]
units2 = units_options[best['units2']]
pool1 = poolsize_option [best['pool1']]
if best['num_layers'] == 1:
    dropout3 = dropout_options[best['dropout3']]
    filter3 = filters_options[best['filter3']]
    kernel3 = kernelsize_option[best['kernel3']]

# 06. Model Training
if best['num_layers'] == 0:
    model = Sequential()
    model.add(Conv1D(filter1, kernel_size=int(kernel1), input_shape=train_x.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(dropout1))
    model.add(MaxPooling1D(pool_size=int(pool1)))
    model.add(Conv1D(filter2, kernel_size=int(kernel2)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout2))
    model.add(Flatten())
    model.add(Dense(units1))
    model.add(Activation('relu'))
    model.add(Dense(units2))
    model.add(Activation('relu'))
    model.add(Dense(dim, activation='softmax'))
else:
    model = Sequential()
    model.add(Conv1D(filter1, kernel_size=int(kernel1), input_shape=train_x.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(dropout1))
    model.add(MaxPooling1D(pool_size=int(pool1)))
    model.add(Conv1D(filter2, kernel_size=int(kernel2)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout2))
    model.add(Conv1D(filter3, kernel_size=int(kernel3)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout3))
    model.add(Flatten())
    model.add(Dense(units1))
    model.add(Activation('relu'))
    model.add(Dense(units2))
    model.add(Activation('relu'))
    model.add(Dense(dim, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = optimizer_option[best['optimizer']], metrics = ['accuracy'])
history = model.fit(train_x, train_y, epochs = 300, batch_size = batch_size, validation_data=(test_x, test_y))

test_scores = model.evaluate(test_x, test_y)
for i, m in enumerate(model.metrics_names):
    print("\n%s: %.3f"% (m, test_scores[i]))
train_scores = model.evaluate(train_x, train_y)
for i, m in enumerate(model.metrics_names):
    print("\n%s: %.3f"% (m, train_scores[i]))
print(train_scores, test_scores)