#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 08:56:24 2019

@author: ayanpoonawala
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt

# import the data set (like csv file or any format file)

dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values  
Y=dataset.iloc[:,1:].values

#fitting simple linear regression

from sklearn.model_selection import train_test_split

#used model_selection in place of cross_validation since the latter is deprecated

X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)

#fitting simple linear reression to the trainig set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# prediting the test set result
y_pred = regressor.predict(X_test)
#visualising the training set results

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary of Experience')
plt.ylabel('Salary')
plt.show()

#visualisinng the test set results
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.ylabel('Salary')
plt.show()