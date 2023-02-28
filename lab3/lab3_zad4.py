# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 08:56:26 2022

@author: maksi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
pd.options.mode.chained_assignment = None


TP = (0,0)  # True Positive
FN = (0,1)  # False Negative
FP = (1,0)  # False Positive
TN = (1,1)  # True Negative

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data.loc[mask, column] = 1
    data.loc[~mask, column] = 0
    return data

def calculate_metrics(cm):
    sensivity = cm[TP] / (cm[TP] + cm[FN])
    precision = cm[TP] / (cm[TP] + cm[FP])
    specificity = cm[TN] / (cm[FP] + cm[TN])
    accuracy = (cm[TP] + cm[TN]) / (cm[TP] + cm[FN] + cm[FP] + cm[TN])
    f1 = (2 * sensivity * precision)/(sensivity + precision)
    return sensivity, precision, specificity, accuracy, f1
    
def printMetrics(a_name, cm, se, p, sp, acc, f1):
    print(f"{a_name}\nArray:\n{cm}\nSensivity: {se}\nPrecision: {p}\nSpecificity: {sp}\nAccuracy: {acc}\nf1: {f1}\n")

data = pd.read_csv("practice_lab_3.csv", sep=";")
col = list(data.columns)

data = qualitative_to_0_1(data, 'Gender', 'Male')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

cat_feature = pd.Categorical(data['Property_Area'])
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])

X = data.drop(columns=('Loan_Status')).values.astype(float)
y = data['Loan_Status'].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

#1

models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))

print("Domyslne parametry")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

#2

models = [kNN(n_neighbors=6, weights="uniform"), SVM(kernel="rbf")]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))

print("Z parametrami innymi niz domyslne")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)