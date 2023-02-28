# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:17:07 2022

@author: maksi
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
pd.options.mode.chained_assignment = None 

TP = (0,0)  # True Positive
FN = (0,1)  # False Negative
FP = (1,0)  # False Positive
TN = (1,1)  # True Negative

def calculate_metrics(cm):
    sensivity = cm[TP] / (cm[TP] + cm[FN])
    precision = cm[TP] / (cm[TP] + cm[FP])
    specificity = cm[TN] / (cm[FP] + cm[TN])
    accuracy = (cm[TP] + cm[TN]) / (cm[TP] + cm[FN] + cm[FP] + cm[TN])
    f1 = (2 * sensivity * precision) / (sensivity + precision)
    return sensivity, precision, specificity, accuracy, f1

def printMetrics(a_name, cm, se, p, sp, acc, f1):
    print(f"{a_name}\nArray:\n{cm}\nSensivity: {se}\nPrecision: {p}\nSpecificity: {sp}\nAccuracy: {acc}\nf1: {f1}\n")

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
col = list(df.columns)

X = df.values
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

dt = DT(max_depth=5)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(40,20))
tree_vis = plot_tree(dt, feature_names=col, class_names=['Y','N'], fontsize=20)
"""
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))
    
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)"""