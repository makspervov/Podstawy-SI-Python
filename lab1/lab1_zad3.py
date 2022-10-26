# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 19:03:16 2022

@author: maksi
"""

import numpy as np
import matplotlib.pyplot as plt

def printPlot(x, y, title=""):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set(xlabel="x", ylabel="y")

x = np.arange(-5, 5, 0.01)

y = np.tanh(x)
printPlot(x, y, "Wykres 1")

y = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) #Alternate form y=tanh(x)
printPlot(x, y, "Wykres 2")

y = 1/(1+np.exp(-x))
printPlot(x, y, "Wykres 3")

y = [0 if i <= 0 else i for i in x]
printPlot(x, y, "Wykres 4")

y = [(np.exp(i) - 1) if i <= 0 else i for i in x]
printPlot(x, y, "Wykres 5")