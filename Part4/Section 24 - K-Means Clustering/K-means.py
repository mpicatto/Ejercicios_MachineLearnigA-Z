#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:09:36 2020

@author: root
"""

#Plantilla de Segmentación (K-means) 

#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Mall_Customers.csv")

X=df.iloc[:,3:5].values

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
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer 
onehotencoder=make_column_transformer((OneHotEncoder(),[0]),remainder="passthrough")
x=onehotencoder.fit_transform(x)
"""    
"""
#dividir dataset en conjunto de entrenamiento y testing


from sklearn.model_selection import train_test_split as splitter
x_train, x_test, y_train,  y_test = splitter(x, y, test_size=0.25, random_state=0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler as scaler
scala_x=scaler()
x_train = scala_x.fit_transform(x_train)
x_test = scala_x.transform(x_test)
"""



#------<AJUSTAR MODELOS DE Segmentacion>----------
#metodo del codo para averiguar el número optimo de clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,
                  init="k-means++",
                  max_iter=300,
                  n_init=10,
                  random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Método del Codo")
plt.xlabel("Número de Clusters")
plt.plot("WCSS(k)")
plt.show()

#Ajustar segmentacion al dataset
means=KMeans(n_clusters=5,
                  init="k-means++",
                  max_iter=300,
                  n_init=10,
                  random_state=0)
y_kmeans=kmeans.fit_predict(X)


#------<VISUALIZAR CLUSTER>----------
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s=100, c="red", label="Cluster 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s=100, c="blue", label="Cluster 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s=100, c="green", label="Cluster 3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s=100, c="cyan", label="Cluster 4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], s=100, c="magenta", label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100, c="yellow", label="baricentros")
plt.title("Clusters de Clientes")
plt.xlabel("Ingresos anuales(en miles de u$s)")
plt.ylabel("Puntuación de Gastos (0-100)")
plt.legend()
plt.show()
