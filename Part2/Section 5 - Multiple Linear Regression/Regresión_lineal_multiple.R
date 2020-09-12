#Regresión Lineal Múltiple
#Plantilla para el Pre Procesado de Datos
#Importar Dataset
df=read.csv('50_Startups.csv')
#df=dataset[,2:3]

#codificar variables categóricas

df$State=factor(df$State,
                   levels=c("New York", "California", "Florida"),
                   labels=c(1,2,3))

#Dividir los datos en entrenamiento y conjunto de test
#install.packages("caTools")
library(caTools)
set.seed(123)
sample=sample.split(df$Profit, SplitRatio = 0.8)
training_set=subset(df, sample==TRUE)
testing_set=subset(df, sample==FALSE)

#Ajustar el modelo de Regresión Lineal Múltiple con el conjunto de entrenamiento
regression=lm(formula = Profit ~.,
              data=training_set)
#predecir resultados con testing dataset
y_pred=predict(regression,newdata=testing_set)

#construir modelo optimo  cpn la Eliminación Hacia Atrás (manual)
sl = 0.05
regression=lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
              data=df)
summary(regression)

regression=lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend , 
              data=df)
summary(regression)

regression=lm(formula = Profit ~ R.D.Spend + Marketing.Spend , 
              data=df)
summary(regression)
regression=lm(formula = Profit ~ R.D.Spend, 
              data=df)
summary(regression)  

# #construir modelo optimo  cpn la Eliminación Hacia Atrás (auto)
# backwardElimination <- function(x, sl) {
#   numVars = length(x)
#   for (i in c(1:numVars)){
#     regressor = lm(formula = Profit ~ ., data = x)
#     maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
#     if (maxVar > sl){
#       j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
#       x = x[, -j]
#     }
#     numVars = numVars - 1
#   }
#   return(summary(regressor))
# }
# 
# sl = 0.05
# dataset = df[, c(1,2,3,4)]
# backwardElimination(training_set, sl)


