# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:05:58 2021

@author: Pramod
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import neighbors

lab_enc = preprocessing.LabelEncoder()

dataset = pd.read_excel('Training Set.xlsx')

dataset_test = pd.read_excel('Test Set.xlsx')
new_input=dataset_test.iloc[:, 1: 12].values

X = dataset.iloc[:, 1: 12].values

y = dataset.iloc[:,12].values

y=lab_enc.fit_transform(y)

model = RandomForestRegressor(n_estimators = 200, random_state = 0)

model.fit(X,y)
print("Fit")
new_output = model.predict(new_input)
print("Predict")

print(new_input,new_output)
df = pd.DataFrame(new_input,new_output)

# print(df)
df.to_excel("output_rf.xlsx")



