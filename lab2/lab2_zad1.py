# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:47:22 2022

@author: maksi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def printScatterPlot(x, y, xl="", yl="", title=""):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set(xlabel=xl, ylabel=yl)

data = pd.read_csv("practice_lab_2.csv", sep=";")
col = data.columns.to_list()
val = data.values

corr_arr = data.corr()

num_col = np.shape(val)[1]

for i in range(num_col - 1):
    printScatterPlot(val[:, i], val[:, -1], col[i], col[-1])