#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:19:30 2020

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:04:51 2020

@author: root
"""
#K-fold Cross Validation & Grid Search Optimization

#Plantilla de Clasificacion Support Vector Machine (kernels)

#------<PREPROCESADO DE DATOS>----------

#Como importar una libreria

import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar dataset
df=pd.read_csv("Social_Network_Ads.csv")

x=df.iloc[:, 2:4].values
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
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer 
onehotencoder=make_column_transformer((OneHotEncoder(),[0]),remainder="passthrough")
x=onehotencoder.fit_transform(x)
"""
#dividir dataset en conjunto de entrenamiento y testing


from sklearn.model_selection import train_test_split as splitter
x_train, x_test, y_train,  y_test = splitter(x, y, test_size=0.25, random_state=0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler as scaler
scala_x=scaler()
x_train = scala_x.fit_transform(x_train)
x_test = scala_x.transform(x_test)

#------<AJUSTAR MODELOS DE CLASIFICACION>----------

#Ajustar regresion con el el conjunto de entrenamiento

#Crear modelo de clasificación aqui 

from sklearn.svm import SVC
classifier=SVC(kernel="rbf",
               random_state=0)
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
#------<APLICAR KFOLD CROSS VALIDATION>------
from sklearn.model_selection import cross_val_score as kfval
accuracies=kfval(estimator =classifier, X=x_train,
                 y=y_train, cv=10)
accuracies.mean()
accuracies.std()

#------<Aplicar Gried Search Optimization>------
from sklearn.model_selection import GridSearchCV as GSCV
parameters=[{"C":[1,10,100,1000],"kernel":["linear"]},
            {"C":[1,10,100,1000],"kernel":["rbf"], "gamma":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]}
            ]
grid_search=GSCV(estimator=classifier,
                 param_grid=parameters,
                 scoring="accuracy",
                 cv=10,
                 n_jobs=-1)

grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

#-------<Clasificar con valores optimos>----

classifier=SVC(C=1,
               kernel="rbf",
               gamma=0.7,
               random_state=0)
classifier.fit(x_train,y_train)
print(classifier)
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


#------<VISUALIZAR MODELOS DE CLASIFICACION>----------

# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM(Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

