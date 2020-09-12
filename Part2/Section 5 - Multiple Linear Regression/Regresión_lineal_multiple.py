#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:29:49 2019

@author: root
"""
#regresión lineal múltiple

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar datase
df=pd.read_csv("50_Startups.csv") 

x=df.iloc[:,:-1].values
y=df.iloc[:, 4].values

#datos  categóricos no ordinales
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer 
onehotencoder=make_column_transformer((OneHotEncoder(),[3]),remainder="passthrough")
x=onehotencoder.fit_transform(x)

#evitar la trampa de las variables ficticias
x=x[:,1:]

#dividir dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split as splitter
x_train, x_test, y_train,  y_test = splitter(x, y, test_size=0.2, random_state=0)

#Modelo de Regresión de entrenamiento
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train, y_train)

#prediccion de resultado en test
y_pred=regression.predict(x_test)

#eliminar variables hacia atras para optimizar el modelo (manual)
"""import statsmodels.regression.linear_model as stat
x=np.append(arr =np.ones((50,1)).astype(int), values =x, axis=1)
sl=0.05
x_opt=x[:,[0,1,2,3,4,5]].tolist()
regression_ols=stat.OLS(endog=y, exog=x_opt).fit()
print(regression_ols.summary())

x_opt=x[:,[0,2,3,4,5]].tolist()
regression_ols=stat.OLS(endog=y, exog=x_opt).fit()
print(regression_ols.summary())
  
x_opt=x[:,[0,3,4,5]].tolist()
regression_ols=stat.OLS(endog=y, exog=x_opt).fit()
print(regression_ols.summary())
"""
#eliminar variables para optimizar el modelo con P valor y R Cuadrado
"""
import statsmodels.regression.linear_model as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()      
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    print(regressor_OLS.summary())    
    return x 
x=np.append(arr =np.ones((50,1)).astype(int), values =x, axis=1) 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]].tolist()
x_Modeled = backwardElimination(x_opt, SL)
"""    

