#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:36:12 2020

@author: root
"""

#Muestreo de Thompson

#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Ads_CTR_Optimisation.csv")

#X=df.iloc[:,3:5].values
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
#Algoritmo de Thompson
import random
N=10000
d=10
number_of_rewards1=[0]*d
number_of_rewards0=[0]*d
total_reward=0
ads_selected=[]
for n in range(0,N):
    max_random=0
    ad=0
    for i in range(0,d):
        random_beta=random.betavariate(number_of_rewards1[i]+1,number_of_rewards0[i]+1)            
        if random_beta > max_random:
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=df.values[n,ad]
    if reward==1:
        number_of_rewards1[ad]=number_of_rewards1[ad]+1
    else:
        number_of_rewards0[ad]=number_of_rewards0[ad]+1
    total_reward=total_reward+reward
    
#Visualizar los Resultados
plt.hist(ads_selected)
plt.title("Histogramas de anuncios")
plt.xlabel("id del anuncio")
plt.ylabel("Frecuencia de visualizacion del anuncio")
plt.show()
