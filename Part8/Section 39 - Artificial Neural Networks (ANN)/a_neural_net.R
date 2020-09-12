#Artificial Neural Network

#Plantilla de Regresion Logística

#--->>Pre Procesado de Datos

#Importar Dataset
df=read.csv('Churn_Modelling.csv')

#filtrado de datasets (filas, columnas)
df=df[,4:14]


#tratamiento de los n/a
# df$Age=ifelse(is.na(df$Age),
#               ave(df$Age,FUN=function(x)mean(x,na.rm = TRUE)),
#               df$Age )
# df$Salary=ifelse(is.na(df$Salary),
#                  ave(df$Salary,FUN=function(x)mean(x,na.rm = TRUE)),
#                  df$Salary )


#codificar variables categóricas a factor y pasar factores a datos numéticos

df$Geography=as.numeric(factor(df$Geography,
                  levels=c("France", "Spain", "Germany"),
                  labels=c(1,2,3)))
df$Gender=as.numeric(factor(df$Gender,
                  levels=c("Female", "Male"),
                  labels=c(0,1)))


#Dividir los datos en entrenamiento y conjunto de test
#install.packages("caTools")
library(caTools)
set.seed(123)
sample=sample.split(df$Exited, SplitRatio = 0.8)
training_set=subset(df, sample==TRUE)
testing_set=subset(df, sample==FALSE)

#Escalado de Valores
training_set[,1:10]=scale(training_set[,1:10])
testing_set[,1:10]=scale(testing_set[,1:10])

#-----<Crear Red Neuronal>-----
library("h2o")
h2o.init(nthreads=-1)
classifier=h2o.deeplearning(y="Exited",
                            training_frame = as.h2o(training_set),
                            activation = "Rectifier",
                            hidden = c(6,6),
                            epochs = 100,
                            train_samples_per_iteration = -2)


#-----<Crear Modelo de Clasificación>----
#crear prediccion si el resultado esperado es una  probabilidad:
prob_pred=h2o.predict(classifier,
                      newdata = as.h2o(testing_set[,-11]))
y_pred=(prob_pred>0.5)
y_pred=as.vector(y_pred)

#Matriz de confucción para evaluar eficiencia 
cm=table(testing_set[,11],
         y_pred)
#Cerrar la sesión de H2o
h2o.shutdown()
