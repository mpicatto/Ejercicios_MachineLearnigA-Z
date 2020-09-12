#Plantilla para el Pre Procesado de Datos
#Importar Dataset
df=read.csv('Data.csv')
#filtrado de datasets (filas, columnas)
#df=dataset[,2:3]
#tratamiento de los n/a
# df$Age=ifelse(is.na(df$Age),
#               ave(df$Age,FUN=function(x)mean(x,na.rm = TRUE)),
#               df$Age )
# df$Salary=ifelse(is.na(df$Salary),
#                  ave(df$Salary,FUN=function(x)mean(x,na.rm = TRUE)),
#                  df$Salary )
#codificar variables categóricas

# df$Country=factor(df$Country,
#                   levels=c("France", "Spain", "Germany"),
#                   labels=c(1,2,3))
# df$Purchased=factor(df$Purchased,
#                   levels=c("No", "Yes"),
#                   labels=c(0,1))
#Dividir los datos en entrenamiento y conjunto de test
#install.packages("caTools")
library(caTools)
set.seed(123)
sample=sample.split(df$Purchased, SplitRatio = 0.8)
training_set=subset(df, sample==TRUE)
testing_set=subset(df, sample==FALSE)

#Escalado de Valores
# training_set[,2:3]=scale(training_set[,2:3])
# testing_set[,2:3]=scale(testing_set[,2:3])  










