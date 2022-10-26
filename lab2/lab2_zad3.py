"""
Created on Wed Oct 26 16:47:44 2022

@author: maksi
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

data = pd.read_csv("practice_lab_2.csv", sep=";")
col = data.columns.to_list()
val = data.values

# procedura usuwania warotści odstających
def remove_outliers(X_train, y_train):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3   
    y_train_no_outliers = y_train[~outliers] 
    X_train_no_outliers = X_train[~outliers,:]
    return X_train_no_outliers, y_train_no_outliers

# procedura zastępowania warotści odstających
def replace_outliers_by_mean(y_train):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3
    y_train_mean = y_train.copy()
    y_train_mean[outliers] = y_train.mean()
    return y_train_mean

def testRegression(X, y, n):
    res = np.zeros(n)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        X_train, y_train = remove_outliers(X_train, y_train)
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        res[i] = mean_absolute_percentage_error(y_test, y_pred)
    return res.mean()

X, y = val[:,:-1], val[:,-1]

print(testRegression(X, y, 100))