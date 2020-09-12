#--->>Pre Procesado de Datos

#Importar Dataset
df=read.csv("Position_Salaries.csv")
#filtrado de datasets (filas, columnas)
df=df[,2:3]


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
#library(caTools)
#set.seed(123)
#sample=sample.split(df$Purchased, SplitRatio = 0.8)
#training_set=subset(df, sample==TRUE)
#testing_set=subset(df, sample==FALSE)

#Escalado de Valores
# training_set[,2:3]=scale(training_set[,2:3])
# testing_set[,2:3]=scale(testing_set[,2:3])  

#-----Crear modelo de regresion-------

#Ajustar Modelo de Regresión SVR
library(e1071)
SVR_reg = svm(formula=Salary ~ .,
              data=df,
              type="eps-regression",
              kernel="radia")



#---------Realizar Predicción____________

#prediccion de nuevos resultados con SVR
y_pred_SVR=predict(SVR_reg, newdata = data.frame(Level=6.5))

#------Crear Visualización de los datos________

#visualizacion modelo SVR
library(ggplot2)
ggplot()+ 
  geom_point(aes(x=df$Level, y=df$Salary),
             color="red")+
  geom_line(aes(x=df$Level , y=predict(SVR_reg, newdata=df)),
            color="blue")+
  ggtitle("Predicción SVR del Sueldo")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo (USD)")
