# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:53:28 2022

@author: maksi
"""

import pandas as pd
import numpy as np

data = pd.read_csv("data-lab1.csv", sep = ';')
col = data.columns
val = data.values

arr1 = val[::2,:] - val[1::2,:]   
                      
arr2 = (val - val.mean()) / val.std() 
                  
arr3 = (val - val.mean(axis=0)) / val.std(axis=0)   
    
arr4 = (val.std(axis=0) / val.mean(axis=0))  
           
arr5 = arr4.max()    
                         
arr6 = (val[:,::1] > val.mean(axis=0)).sum(axis=0)  
 
mask = (val == val.max())[:,::1].sum(axis=0) > 0        
arr7 = np.array(col)[mask]

mask = (val == 0).sum(axis=0) == (val == 0).sum(axis=0).max()
arr8 = np.array(col)[mask]

mask = val[::2,:].sum(axis=0) > val[1::2,:].sum(axis=0) 
arr9 = np.array(col)[mask]