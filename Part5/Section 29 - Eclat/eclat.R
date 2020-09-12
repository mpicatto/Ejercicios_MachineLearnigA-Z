#Associatopn Rule Learning (Eclat)

#--->>Pre Procesado de Datos

#Importar Dataset
library(arules)
df=read.csv('Market_Basket_Optimisation.csv',header = FALSE)
# #filtrado de datasets (filas, columnas)
# df=df[,3:5]
#Crear Sparce Matrix
df=read.transactions("Market_Basket_Optimisation.csv",
                     sep = ",",
                     rm.duplicates = TRUE)
# Histograma de Transacciones
itemFrequencyPlot(df,topN=20)

#Entrenar Algoritmo con el Dataset
rules=eclat(data = df, parameter = list(support=0.01, minlen=2))

#Visualizacion de los resultados
inspect(sort(rules, by='support')[1:10])



