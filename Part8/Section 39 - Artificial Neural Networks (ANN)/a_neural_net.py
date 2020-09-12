#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:25:17 2020

@author: root
"""

#Aritificial Neural Network

#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Churn_Modelling.csv")

X=df.iloc[:, 3:13].values
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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer([("Geography", OneHotEncoder(), [1])], # The column numbers to be transformed (here is [1])
                                  remainder="passthrough") # Leave the rest of the columns untouched
X = onehotencoder.fit_transform(X)
X = X[:, 1:]
#dividir dataset en conjunto de entrenamiento y testing


from sklearn.model_selection import train_test_split as splitter
x_train, x_test, y_train,  y_test = splitter(X, y, test_size=0.2, random_state=0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler as scaler
scala_x=scaler()
x_train = scala_x.fit_transform(x_train)
x_test = scala_x.transform(x_test)

#------<AJUSTAR MODELOS DE CLASIFICACION>----------
"""
#Ajustar regresion con el el conjunto de entrenamiento

#Crear modelo de clasificación aqui 

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
print(classifier)


#predicción de nuestros modelos con conjunto de test

y_pred=classifier.predict(x_test)

#predicción de nuestros modelos con nuevo valor
#y_pred=clasifier.predict(6.5)

#crear matriz de confusión (evaluar prediccion)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
"""