# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:53:40 2022

@author: maksi
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

ionosphere=pd.read_csv('ionosphere_data.csv', header=None)
ionosphere.columns=['C'+str(i) for i in range(36)]

ionosphere.drop('C0', axis=1, inplace=True)
print(ionosphere.head())
print(ionosphere.shape)
print(ionosphere.iloc[:,-1].value_counts())


X, y = ionosphere.iloc[:,:-1], ionosphere.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2022, stratify=y)

pca_transform=PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances=variances.cumsum()
plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.show()

pca95=PCA(n_components=0.95)
X_train_pca=pca95.fit_transform(X_train)

PC_num = (cumulated_variances<0.95).sum()+1
print("Aby wyjaśnić 95% wariancji, potrzeba "+str(PC_num)+' składowych głównych')

print("\n1) PCA, StandartScaler, kNN")
pipe=Pipeline([
    ['transformer', PCA(0.95)],
    ['scaler', StandardScaler()],
    ['classifier', kNN(weights='distance')]
    ])
pipe.fit(X_train, y_train)
y_pred=pipe.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print("\n2) FastICA, MinMaxScaler, kNN")
pipe2=Pipeline([
    ['transformer', FastICA(20, random_state=2022)],
    ['scaler', MinMaxScaler()],
    ['classifier', kNN(weights='distance')]
    ])
pipe2.fit(X_train, y_train)
y_pred=pipe2.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print("\n3) PCA, RobustScaler, SVM")
pipe3=Pipeline([
    ['transformer', PCA(0.95)],
    ['scaler', RobustScaler()],
    ['classifier', SVM()]
    ])
pipe3.fit(X_train, y_train)
y_pred=pipe3.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print("\n4) FastICA, *brak*, SVM")
pipe4=Pipeline([
    ['transformer', FastICA(20, random_state=2022)],
    ['classifier', SVM()]
    ])
pipe4.fit(X_train, y_train)
y_pred=pipe4.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print("\n5) PCA, *brak*, SVM")
pipe4=Pipeline([
    ['transformer', PCA(0.95)],
    ['classifier', SVM()]
    ])
pipe4.fit(X_train, y_train)
y_pred=pipe4.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


aux=X_train_pca.copy()
aux=pd.DataFrame(aux)
aux.columns=['PC'+str(i+1) for i in range(23)]
aux['target']=y_train
sns.lmplot(x='PC1', y='PC2', hue='target',data=aux, fit_reg=False)
plt.show()

