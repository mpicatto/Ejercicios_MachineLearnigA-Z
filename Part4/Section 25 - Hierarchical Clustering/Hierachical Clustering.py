#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:36:47 2020

@author: root
"""

# Segmentación Jerarqioca (Hierarchical Clustering)
#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Mall_Customers.csv")

X=df.iloc[:,3:5].values
"""
#tratamiento de los nan
from sklearn.impute import SimpleImputer as Imputer
imputer=Imputer(missing_values= np.nan, strategy="mean")
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#datos categóricos ordinales codificar 
from sklearn import preprocessing as pre
le_y=pre.LabelEncoder()
y=le_y.fit_transform(y)

#datos  categóricos no ordinales

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer 
onehotencoder=make_column_transformer((OneHotEncoder(),[0]),remainder="passthrough")
x=onehotencoder.fit_transform(x)
"""
#------<AJUSTAR MODELOS DE Segmentacion>----------
#Crear dendrograma para encontrar numero optimo de clústers
import scipy.cluster.hierarchy as sch
dengrograma=sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

#Ajustar segmentacion al dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, 
                           affinity="euclidean",
                           linkage="ward")
y_hc=hc.fit_predict(X)

#------<VISUALIZAR CLUSTER>----------

plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s=100, c="red", label="Cautos")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s=100, c="blue", label="Estandard")
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s=100, c="green", label="Objetivos")
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s=100, c="cyan", label="Descuidades")
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s=100, c="magenta", label="Conservadores")
plt.title("Clusters de Clientes")
plt.xlabel("Ingresos anuales(en miles de u$s)")
plt.ylabel("Puntuación de Gastos (0-100)")
plt.legend()
plt.show()