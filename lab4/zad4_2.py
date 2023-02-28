#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#podłączamy biblioteki do programu
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from scipy.fft import fft

#funkcja przekształcenia wartości w polu Sex na wartości logiczne
def qualitative_to_0_1(data, column, value_to_be_1):
     mask = data[column].values == value_to_be_1
     data[column][mask] = 1
     data[column][~mask] = 0
     return data
 
#odczyt danych z pliku
data = pd.read_csv("voice_extracted_features.csv", sep=',')
data = qualitative_to_0_1(data, 'label', 'female')
col = list(data.columns)
val = data.values.astype(float)

X = val[:,:-1]
y = val[:,-1]

#funkcja skalowania danych
def scale(X_train, X_test):
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)
     return X_train, X_test
#funkcja wyszukiwania i generacji macierzy pomyłek
def meanConfusionMatrix(X, y, model, n):
     res = np.zeros((2,2))
     args = [['transformer', PCA(9)],
             ['scaler', StandardScaler()],
             ['classifier', model]]
     for i in range(n):
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
         X_train, X_test = scale(X_train, X_test)
         pipe = Pipeline(args)
         pipe.fit(X_train, y_train)
         y_pred = pipe.predict(X_test)
         res += confusion_matrix(y_test, y_pred)
     return res / n
#testowanie algorytmu kNN
print("kNN: ")
print(meanConfusionMatrix(X, y, kNN(n_neighbors=5, weights='distance'), 30))
#testowanie algorytmu SVM
print("SVM: ")
print(meanConfusionMatrix(X, y, SVM(), 30))
#testowanie algorytmu Decision Tree
print("DT: ")
print(meanConfusionMatrix(X, y, DT(), 30))