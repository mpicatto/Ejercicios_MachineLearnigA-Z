#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:52:41 2020

@author: root
"""


#Natural Lenguage Processing

#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

#limpieza de texto
import re
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub("[^a-zA-Z]"," ",
                  df['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)
        
#Crear Bolsa de Palabras (Bag of Words)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,-1].values


from sklearn.model_selection import train_test_split as splitter
x_train, x_test, y_train,  y_test = splitter(X, y, test_size=0.20, random_state=0)


#------<AJUSTAR MODELOS DE CLASIFICACION>----------

#Ajustar regresion con el el conjunto de entrenamiento

#Crear modelo de clasificación aqui 

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
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