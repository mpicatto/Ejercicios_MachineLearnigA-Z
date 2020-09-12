#Aplicacion K-Fold Cross Validation

#Plantilla de Clasificacion Support Vector Machine (kernels)

#--->>Pre Procesado de Datos

#Importar Dataset
df=read.csv('Social_Network_Ads.csv')
# 
# #filtrado de datasets (filas, columnas)
df=df[,3:5]


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


# #Dividir los datos en entrenamiento y conjunto de test
# #install.packages("caTools")
library(caTools)
set.seed(123)
sample=sample.split(df$Purchased, SplitRatio = 0.75)
training_set=subset(df, sample==TRUE)
testing_set=subset(df, sample==FALSE)
# 
# #Escalado de Valores
training_set[,1:2]=scale(training_set[,1:2])
testing_set[,1:2]=scale(testing_set[,1:2])
# 
# #-----Crear modelo de Clasificacion-------
# 
# #Ajustar Modelo de Clasificacion con conjunto de training
library(e1071) 

# # ------Aplicar K fold Validation------
library(caret)
folds=createFolds(training_set$Purchased, k=10)
cv=lapply(
  folds, function(x){
    training_fold=training_set[-x, ]
    testing_fold=testing_set[x,]
    classifier = svm(formula = Purchased ~ .,
                     data = training_fold,
                     type="C-classification",
                     kernel="radial")
    y_pred=predict(classifier, newdata = testing_fold[,-3])
    cm=table(testing_fold[,3],
             y_pred)
    accuracy =(cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
    return(accuracy)
    
  })



# #------Crear Visualización de los datos-----------
# Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')

y_grid =predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualización del conjunto de testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')

y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))