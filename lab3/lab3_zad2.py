# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 08:31:33 2022

@author: maksi
"""

import pandas as pd

data = pd.read_csv('practice_lab_3.csv', sep=';')
columns = list(data.columns)

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

data = qualitative_to_0_1(data, 'Gender', 'Male')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

cat_feature = pd.Categorical(data['Property_Area'])
one_hot = pd.get_dummies(cat_feature)

data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])