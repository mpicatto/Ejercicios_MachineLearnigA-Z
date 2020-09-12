#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:21:13 2020

@author: root
"""

#Regresión con Arboles de Decision

#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Position_Salaries.csv")

x=df.iloc[:, 1:2].values
y=df.iloc[:,-1].values
y=y.reshape(-1,1)
"""
#Escalado de variables
from sklearn.preprocessing import StandardScaler as scaler
scala_x=scaler()
x_train = scala_x.fit_transform(x_train)
x_test = scala_x.transform(x_test)
"""
#------<AJUSTAR MODELOS DE REGRESION>----------

#Ajustar regresion con Aroboles de Decision
from sklearn.tree import DecisionTreeRegressor
regression=DecisionTreeRegressor(random_state=0)
regression.fit(x,y)
print(regression)

#predicción de nuestros modelos

y_pred = regression.predict([[6.5]])


#------<VISUALIZAR MODELOS DE REGRESION>----------

#visualizacion de modelo sin suavisado
plt.scatter(x,y,color="red")
plt.plot(x,regression.predict(x), color="blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del Empleado")
plt.ylabel("Sueldo (en u$s)")
plt.show()

"""
#visualizacion del modelo con suavizado
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color="red")
plt.plot(x_grid,regression.predict(x_poli), color="blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del Empleado")
plt.ylabel("Sueldo (en u$s)")
plt.show()
"""
