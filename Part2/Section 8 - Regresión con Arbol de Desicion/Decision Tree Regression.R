# Regression con Arbol de Decision

#--->>Pre Procesado de Datos

#Importar Dataset
df=read.csv("Position_Salaries.csv")
#filtrado de datasets (filas, columnas)
df=df[,2:3]

#-----Crear modelo de regresion-------

#Ajustar Modelo de regresion con Arbol de desicion
#Crear variables de regresion
library(rpart)
regression = rpart(formula=Salary ~ .,
              data=df,
              control = rpart.control(minsplit = 1))



#---------Realizar Predicción____________

#prediccion de nuevos resultados con Arbol de desicion
y_pred=predict(regression, newdata = data.frame(Level=6.5))

#------Crear Visualización de los datos________

#visualizacion modelo de Desicion
#visualizacion modelo de sin suavizado
library(ggplot2)
ggplot()+ 
  geom_point(aes(x=df$Level, y=df$Salary),
             color="red")+
  geom_line(aes(x=df$Level , y=predict(regression, newdata=data.frame(Level =df$Level))),
            color="blue")+
  ggtitle("Predicción con Arbol de Decision del Sueldo")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo (USD)")


#visualizacion modelo de con suavizado

x_grid=seq(min(df$Level),max(df$Level),0.1)
ggplot()+ 
  geom_point(aes(x=df$Level, y=df$Salary),
             color="red")+
  geom_line(aes(x=x_grid , y=predict(regression, 
                                     newdata=data.frame(Level=x_grid))),
                                      color="blue")+
  ggtitle("Predicción con Arbol de Decision del Sueldo")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo (USD)")

