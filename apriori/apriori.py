#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on sat Jan 23 08:47:55 2019

@author: ayan poonawala
"""

#importing the library
import numpy as np     #maths function
import matplotlib as plt #plotting diagram
import pandas as pd   #file manage


# importimg  the data set
#read file help of pandas
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None) 
transactions = [] #list
for i in range(0,7501): #range start from 0 to  max
        transactions.append([str(dataset.values[i,j]) for j in range(0,20)]) #list comprehension
#Training apyori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003,min_confidence = 0.2, min_lift = 3, min_length = 2) #apiori aldo

#visualizing results

results = list(rules)#convert in list
results_list = []
for i in range(1,len(results)):
    results_list.append('RULE:\t' + str (results[i][0]).replace('frozenset','') +
    'n\SUPPORT:\t' + str (results[i][1]) +
    'n\CONF:\t' + str (results[i][2][0][2]) + 
    'n\LIFT:\t' + str (results[i][2][0][3]))

''' 
support(X)=total no of transc/total
#lift=support(x union y)/support(x)* support(y)
if x>y
'''
    