# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 09:27:50 2022

@author: maksi
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.datasets import load_diabetes

def replace_outliers_by_mean(y_train):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3
    y_train_mean = y_train.copy()
    y_train_mean[outliers] = y_train.mean()
    return y_train_mean

def test_regression(X, y, n):
    res = np.zeros(n)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        y_train = replace_outliers_by_mean(y_train)
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        res[i] = mean_absolute_percentage_error(y_test, y_pred)
    return res.mean()

data = load_diabetes()
dane = pd.DataFrame(data.data, columns = data.feature_names)
col = dane.columns.to_list()
X = dane.values   
dane['target'] = data.target

print(test_regression(X, dane['target'], 100))