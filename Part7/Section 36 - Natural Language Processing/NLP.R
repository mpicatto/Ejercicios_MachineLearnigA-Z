#Natural Lenguage Processing
#--->>Pre Procesado de Datos

#Importar Dataset

df_original=read.delim("Restaurant_Reviews.tsv",quote = '',stringsAsFactors = FALSE)

#----Limpieza del texto----
#install.packages("SnowballC")
#install.packages("tm")
library(tm)
library(SnowballC)
corpus=VCorpus(VectorSource(df_original$Review))
#convertir texto a solo mminusculas
corpus=tm_map(corpus,content_transformer(tolower))
#consltar resultado as.character(corpus[[1]])
#Eliminar Números
corpus=tm_map(corpus,removeNumbers)
#Eliminar Signos de Puntuación
corpus=tm_map(corpus,removePunctuation)
#Eliminar palabras no relevantes
corpus=tm_map(corpus,removeWords,stopwords(kind = "en"))
#Eliminar palabras derivadas
corpus=tm_map(corpus,stemDocument)
#Eliminar espacios extra
corpus=tm_map(corpus,stripWhitespace)

#Crear el modelo de Bag of Words
dtm=DocumentTermMatrix(corpus)
dtm=removeSparseTerms(dtm,0.999)

#----Aplicar Modelor de Clasificación----
#Random Forest

#Importar Dataset
df=as.data.frame(as.matrix(dtm))
df$Liked=df_original$Liked

# Codificar la variable de clasificación como factor
df$Liked = factor(df$Liked, levels = c(0,1))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(df$Liked, SplitRatio = 0.80)
training_set = subset(df, split == TRUE)
testing_set = subset(df, split == FALSE)

# Ajustar el Random Forest con el conjunto de entrenamiento.
#install.packages("randomForest")
library(randomForest)
classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[, 692], y_pred)
