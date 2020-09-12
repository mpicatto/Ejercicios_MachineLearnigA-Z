#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:14:17 2019

@author: root
"""

#regresion polinomica

#Plantilla de Preprocesado

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar datase
df=pd.read_csv("Position_Salaries.csv")

x=df.iloc[:, 1:2].values
y=df.iloc[:,-1].values


#Ajustar regresion lineal con el dataset 

from sklearn.linear_model import LinearRegression
linearR=LinearRegression()
linearR.fit(x,y)
print(linearR)
#Ajustar regerasion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures as poli
poliR=poli(degree=2)
x_poli=poliR.fit_transform(x)
linearR2=LinearRegression()
linearR2.fit(x_poli,y)
print(linearR2)

#visualizacion de modelo lineal
plt.scatter(x,y,color="red")
plt.plot(x,linearR.predict(x), color="blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del Empleado")
plt.ylabel("Sueldo (en u$s)")
plt.show()


#visualizacion del modelo polinimico 
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color="red")
plt.plot(x,linearR2.predict(x_poli), color="blue")
plt.title("Modelo de Regresión Polinomica")
plt.xlabel("Posición del Empleado")
plt.ylabel("Sueldo (en u$s)")
plt.show()

#predicción de nuestros models
print(linearR.predict([[6.5]]))
linearR2.predict(poliR.fit_transform(6.5)