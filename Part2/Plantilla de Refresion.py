#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:32:11 2019

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:14:17 2019

@author: root
"""

#Plantilla de Regresion

#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Position_Salaries.csv")

x=df.iloc[:, 1:2].values
y=df.iloc[:,-1].values

#tratamiento de los nan
"""from sklearn.impute import SimpleImputer as Imputer
imputer=Imputer(missing_values= np.nan, strategy="mean")
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])"""

#datos categóricos ordinales codificar 
"""from sklearn import preprocessing as pre
le_y=pre.LabelEncoder()
y=le_y.fit_transform(y)"""

#datos  categóricos no ordinales
"""from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer 
onehotencoder=make_column_transformer((OneHotEncoder(),[0]),remainder="passthrough")
x=onehotencoder.fit_transform(x)"""

"""
#dividir dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split as splitter
x_train, x_test, y_train,  y_test = splitter(x, y, test_size=0.2, random_state=0)
"""
"""
#Escalado de variables
from sklearn.preprocessing import StandardScaler as scaler
scala_x=scaler()
x_train = scala_x.fit_transform(x_train)
x_test = scala_x.transform(x_test)
"""
#------<AJUSTAR MODELOS DE REGRESION>----------

#Ajustar regresion lineal con el dataset 

"""
from sklearn.linear_model import LinearRegression
linearR=LinearRegression()
linearR.fit(x,y)
print(linearR)
"""
"""
#Ajustar regerasion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures as poli
poliR=poli(degree=2)
x_poli=poliR.fit_transform(x)
linearR2=LinearRegression()
linearR2.fit(x_poli,y)
print(linearR2)
"""
#predicción de nuestros modelos
"""
y_pred=regression.predict(6.5)
"""

#------<VISUALIZAR MODELOS DE REGRESION>----------

#visualizacion de modelo sin suavizado
"""
plt.scatter(x,y,color="red")
plt.plot(x,regression.predict(x), color="blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del Empleado")
plt.ylabel("Sueldo (en u$s)")
plt.show()
"""
"""
#visualizacion del modelo con suavizado
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color="red")
plt.plot(x,regression.predict(x_poli), color="blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del Empleado")
plt.ylabel("Sueldo (en u$s)")
plt.show()
"""
