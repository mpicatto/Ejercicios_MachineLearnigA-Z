#Clustering con Hierachical Clustering
#Importar los Datos
df=read.csv("Mall_Customers.csv")
# filtrado de datasets (filas, columnas)
X=df[,4:5]
#dendrograma calcular K optimo
library(stats)
dendrogram=hclust(dist(X, method = "euclidean"),
                  method ="ward.D")
plot(dendrogram,
     main = "Dendrograma",
     xlab = "Clientes del Centro Comercial",
     ylab = "Distancia Eucliedea")

#Aplicar  modelo con K optimo 

hc=hclust(dist(X, method = "euclidean"),
                  method ="ward.D")
y_hc=cutree(hc,k = 5)

#Visualizacion de los Cluesters
library("cluster")
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 4,
         plotchar = TRUE,
         span = TRUE,
         main = "Cluster de Clientes",
         xlab = "Ingresos Anuales",
         ylab = "Puntuación(1-100)")
