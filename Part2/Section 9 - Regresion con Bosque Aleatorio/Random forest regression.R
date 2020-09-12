# Regression con Random Forest

#--->>Pre Procesado de Datos

#Importar Dataset
df=read.csv("Position_Salaries.csv")
#filtrado de datasets (filas, columnas)
df=df[,2:3]

#-----Crear modelo de regresion-------

#Ajustar Modelo de regresion con Arbol de desicion
#Crear variables de regresion
install.packages("randomForest")
library(randomForest)
set.seed(1234)
regression = randomForest(x=df[1],
                          y=df$Salary,
                          ntree=550)



#---------Realizar Predicción____________

#prediccion de nuevos resultados con Arbol de desicion
y_pred=predict(regression, newdata = data.frame(Level=6.5))

#------Crear Visualización de los datos________

#visualizacion modelo de Desicion
library(ggplot2)

#visualizacion modelo de sin suavizado

ggplot()+ 
  geom_point(aes(x=df$Level, y=df$Salary),
             color="red")+
  geom_line(aes(x=df$Level , y=predict(regression, newdata=data.frame(Level =df$Level))),
            color="blue")+
  ggtitle("Predicción con Random Forest")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo (USD)")


#visualizacion modelo de con suavizado

x_grid=seq(min(df$Level),max(df$Level),0.01)
ggplot()+ 
  geom_point(aes(x=df$Level, y=df$Salary),
             color="red")+
  geom_line(aes(x=x_grid , y=predict(regression, 
                                     newdata=data.frame(Level=x_grid))),
            color="blue")+
  ggtitle("Predicción con Random Forest")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo (USD)")
