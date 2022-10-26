import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def printPlot(x, y, title=""):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set(xlabel="x", ylabel="y")

data = pd.read_csv("data-lab1.csv", sep=";")
col = data.columns
val = data.values

corr_arr = data.corr()  # macierz korelacji

num_col = np.shape(val)[1]  # liczba kolumn tablicy val
for i in range(0, num_col): # generowanie wykresów punktowych 
    for j in range(0, num_col):
        printPlot(val[:,i], val[:,j], col[i] + ' od ' + col[j])