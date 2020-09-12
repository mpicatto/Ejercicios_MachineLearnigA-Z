#Plantilla de Regresion

#--->>Pre Procesado de Datos

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
#library(caTools)
#set.seed(123)
#sample=sample.split(df$Purchased, SplitRatio = 0.8)
#training_set=subset(df, sample==TRUE)
#testing_set=subset(df, sample==FALSE)

#Escalado de Valores
# training_set[,2:3]=scale(training_set[,2:3])
# testing_set[,2:3]=scale(testing_set[,2:3])  

#-----Crear modelo de regresion-------

#Ajustar Modelo de Regresión Lineal con el Dataset
"""
Lin_reg = lm(formula = Salary ~ .,
             data = df)
"""
#Ajustar Modelo de Regresión Polinómica con el Dataset
"""
df$Level2=df$Level ^ 2
df$Level3=df$Level ^ 3
df$Level4=df$Level ^ 4
Poly_reg = lm(formula = Salary ~ .,
              data = df)
"""
#------Crear Visualización de los datos________

#visualizacion modelo lineal
"""
library(ggplot2)
ggplot()+ 
  geom_point(aes(x=df$Level, y=df$Salary),
             color="red")+
  geom_line(aes(x=df$Level , y=predict(Lin_reg, newdata=df)),
            color="blue")+
  ggtitle("Predicción Lineal del Sueldo")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo (USD)")

#visualizacion modelo polinomico
#library(ggplot2)
x_grid=seq(min(df$Level),max(df$Level),0.1)
ggplot()+ 
  geom_point(aes(x=df$Level, y=df$Salary),
             color="red")+
  geom_line(aes(x=x_grid , y=predict(Poly_reg, newdata=data.frame(Level=x_grid,
                                                                  Level2=x_grid ^ 2,
                                                                  Level3=x_grid ^ 3,
                                                                  Level4=x_grid ^ 4))),
            color="blue")+
  ggtitle("Predicción Polinomica del Sueldo")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo (USD)")
"""
#---------Realizar Predicción____________

#prediccion de nuevos resultados con Regresion Lineal
#y_pred_lm=predict(Lin_reg, newdata = data.frame(Level=6.5))
#prediccion de nuevos resultados con Regresión Polinómica
"""
y_pred_pm=predict(Poly_reg, newdata = data.frame(Level=6.5,
                                                 Level2=6.5^2,
                                                 Level3=6.5^3,
                                                 Level4=6.5^4))
"""
