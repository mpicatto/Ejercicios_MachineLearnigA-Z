#Clustering con K means
#Importar los Datos
df=read.csv("Mall_Customers.csv")
# filtrado de datasets (filas, columnas)
X=df[,4:5]
#Metodo del codo
library(stats)
set.seed(6)
wcss =vector()
for (i in 1:10){
wcss[i] <-sum(kmeans(X,i)$withinss)
}
plot(1:10, wcss, type='b', main = "Método del Codo",
     xlab = "Número de Clusteres (k)",ylab = "WCSS(k)")

#Aplicar Kmeans con K optimo 
set.seed(29)
kmeans<-kmeans(X,5,iter.max = 300,nstart = 10)

#Visualizacion de los Cluesters
library("cluster")
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 4,
         plotchar = TRUE,
         span = TRUE,
         main = "Cluster de Clientes",
         xlab = "Ingresos Anuales",
         ylab = "Puntuación(1-100)")
