#Reinforcesd learning (Muestreo de Thomson)
#--->>Pre Procesado de Datos

#Importar Dataset

df=read.csv('Ads_CTR_Optimisation.csv')

# #filtrado de datasets (filas, columnas)
# df=df[,3:5]

#Algoritmo UCB
d=10 #numero de anuncioas
N=10000 #número de casos 
number_of_rewards1=integer(d)
number_of_rewards0=integer(d)
ads_selected=integer(0)
total_reward=0
for(n in 1:N){
  max_random= 0
  ad = 0
  for(i in 1:d){
    random_beta=rbeta(n=1,
                      shape1 = number_of_rewards1[i]+1,
                      shape2 = number_of_rewards0[i]+1)
    if(random_beta > max_random){
      max_random= random_beta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = df[n, ad]
  if (reward==1){
    number_of_rewards1[ad]=number_of_rewards1[ad]+1
  }
  else{
    number_of_rewards0[ad]=number_of_rewards0[ad]+1
  }
  total_reward = total_reward + reward
}
#visualizacion de histograma
hist(ads_selected,
     col = "lightblue",
     main = "Histograma de los Anuncios",
     xlab = "ID del Anuncio",
     ylab = "Frecuencia absoluta del anuncio")
