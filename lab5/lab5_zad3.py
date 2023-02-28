# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:32:36 2022

@author: maksi
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import reciprocal
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

data = load_digits()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
num_epochs=10

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, activation='relu', batch_size=10):
    model = Sequential()
    model.add(Dense(n_neurons, input_shape = (X.shape[1],)))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=('accuracy'))
    return model    

#%%
# GRID SEARCH
keras_classifier=KerasClassifier(build_model)
param_distribs={
    'n_hidden': [0, 1, 2, 3],
    'n_neurons': [30, 50],
    'learning_rate': [0.01, 0.001, 0.0001],
    'activation': ['tanh', 'relu'],
    'batch_size': [10, 14]
    }
grid_search_cv=GridSearchCV(keras_classifier, param_distribs)
grid_search_cv.fit(X_train, y_train, epochs=num_epochs)
best_params_from_grid=grid_search_cv.best_params_
best_model_from_grid=grid_search_cv.best_estimator_

#%%
# RANDOMIZED SEARCH

keras_classifier=KerasClassifier(build_model)
param_distribs={
    'n_hidden': [0,1,2,3],
    'n_neurons': np.arange(1,100),
    'learning_rate': reciprocal(3e-4, 3e-2),
    'activation': ['tanh', 'relu'],
    'batch_size':np.arange(10,12)
    }
rnd_search_cv=RandomizedSearchCV(keras_classifier, param_distribs, n_iter=10, cv=5)
rnd_search_cv.fit(X_train, y_train, epochs=num_epochs)

best_params_from_random=rnd_search_cv.best_params_ 
best_model_from_random=rnd_search_cv.best_estimator_
