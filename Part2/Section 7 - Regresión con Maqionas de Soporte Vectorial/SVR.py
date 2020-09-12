#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:34:31 2020

@author: root
"""

#SVR

#Plantilla de Regresion

#------<PREPROCESADO DE DATOS>----------

# Importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Position_Salaries.csv")

#seleccionar datos para las variables independientes(x) y la y 

x=df.iloc[:, 1:2].values
y=df.iloc[:,-1].values

#Escalado de variables
from sklearn.preprocessing import StandardScaler as scaler
sc_x=scaler()
sc_y=scaler()
x= sc_x.fit_transform(x)
y= sc_y.fit_transform(y.reshape(-1,1))

#------<AJUSTAR MODELOS DE REGRESION>----------

#Ajustar regresion lineal con el dataset 

"""
from sklearn.linear_model import LinearRegression
linearR=LinearRegression()
linearR.fit(x,y)
print(linearR)
"""

#Ajustar regerasion polinomica con el dataset
"""
from sklearn.preprocessing import PolynomialFeatures as poli
poliR=poli(degree=2)
x_poli=poliR.fit_transform(x)
linearR2=LinearRegression()
linearR2.fit(x_poli,y)
print(linearR2)
"""
#Ajustar SVR
from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x,y)


#predicción de nuestros modelos

y_pred = sc_y.inverse_transform(svr_reg.predict(sc_x.transform(np.array([[6.5]])))) 

#------<VISUALIZAR MODELOS DE REGRESION>----------

#visualizacion de modelo lineal
"""
plt.scatter(x,y,color="red")
plt.plot(x,regression.predict(x), color="blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del Empleado")
plt.ylabel("Sueldo (en u$s)")
plt.show()
"""
#visualizacion del modelo polinimico 
"""
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color="red")
plt.plot(x,regression.predict(x_poli), color="blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del Empleado")
plt.ylabel("Sueldo (en u$s)")
plt.show()
"""
#visualizacion de SVR
X_init = np.arange(sc_x.inverse_transform(min(x)),

                   sc_x.inverse_transform(max(x)),

                   0.1)

X_init = X_init.reshape(-1,1)

plt.scatter(sc_x.inverse_transform(x),

            sc_y.inverse_transform(y),

            c='red')

plt.plot(X_init,

         sc_y.inverse_transform(svr_reg.predict(sc_x.transform(X_init))),

         c='blue')

plt.title('Módelo de Regresión SVR')

plt.xlabel('Posición del Empleado')

plt.ylabel('Sueldo en US$')

plt.show()
