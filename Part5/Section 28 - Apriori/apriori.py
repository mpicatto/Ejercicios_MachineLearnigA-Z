#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:29:23 2020

@author: root
"""

#Plantilla de Association Rule Learning -Apriori

#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Market_Basket_Optimisation.csv", header=None)

#X=df.iloc[:,3:5].values

#Crear lista de transacciones

transactions=[]
for i in range(0,7501):
    transactions.append([str(df.values[i,j]) for j in range(0,20)])
    
#entrenar algoritmo apriori
from apyori import apriori    
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2,
                min_lift = 3, min_length = 2)
#visualizar resultados
results=list(rules)
results[0]