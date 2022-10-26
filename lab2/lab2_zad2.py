# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:47:36 2022

@author: maksi
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

data = pd.read_csv("practice_lab_2.csv", sep=";")
col = data.columns.to_list()
val = data.values

def testRegression(X, y, n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)
    y_pred = linReg.predict(X_test)
    print("Średnia: ")
    return mean_absolute_percentage_error(y_test, y_pred)

X, y = val[:,:-1], val[:,-1]

print(testRegression(X, y, 100))