#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:47:17 2019

@author: root
"""
#Plantilla de Preprocesado

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar datase
df=pd.read_csv("Salary_Data.csv")

x=df.iloc[:, :-1].values
y=df.iloc[:, 1].values


#dividir dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split as splitter
x_train, x_test, y_train,  y_test = splitter(x, y, test_size=1/3, random_state=0)
x_train=x_train.reshape(-1,1)
#x_test=x_test.reshape(-1,1)

#Escalado de variables (generalmente nose usa en regresión lineal)

#Modelo de Regresión de entrenamiento
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train, y_train)

#predecir el conjunto de test

y_pred=regression.predict(x_test)

#visualizar resultados de entrenamiento
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train, regression.predict(x_train), color="blue")
plt.title("Sueldo por años de experiencia (training)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en u$s)")
plt.show()
#visualizar prediccion
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train, regression.predict(x_train), color="blue")
plt.title("Sueldo por años de experiencia (test)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en u$s)")
plt.show()