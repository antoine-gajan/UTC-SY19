# Chargement des données de prostate
prostate <- read.csv("prostate.data", sep = "\t")
prostate


# Affichage pour voir quelles variables contribuent à lpsa
plot(lpsa ~lcavol, data = prostate) # Oui, lien linéaire
plot(lpsa ~lweight, data = prostate)
plot(lpsa ~age, data = prostate)
plot(lpsa ~lbph, data = prostate)
boxplot(lpsa ~svi, data = prostate) # Oui
plot(lpsa ~lcp, data = prostate)
boxplot(lpsa ~gleason, data = prostate)
plot(lpsa ~pgg45, data = prostate)

# Installation du package FNN
#install.packages("FNN")
library(FNN)


train_data = prostate[prostate$train, c("lcavol", "lweight", "age", "lbph", "lpsa")]
train_data_X = train_data[c("lcavol", "lweight", "age", "lbph")]
train_data_y = train_data$lpsa

test_data = prostate[!prostate$train, c("lcavol", "lweight", "age", "lbph", "lpsa")]
test_data_X = test_data[c("lcavol", "lweight", "age", "lbph")]
test_data_y = test_data$lpsa

knn.reg(train = train_data_X, test = test_data_X, y = train_data_y)$pred

# Représentation graphique du MSE sur l'ensemble d'apprentissage
mse = function(actual, predicted) {
  mean((actual - predicted) ^ 2)
}

liste_voisins <- 1:20
liste_erreurs_train <- c()
for (nb_voisins in liste_voisins) {
  knn_values <- knn.reg(train = train_data_X, test = test_data_X, y = train_data_y, k = nb_voisins)
  MSE = mse(test_data_y, knn_values$pred)
  liste_erreurs_train <- c(liste_erreurs_train, MSE)
} 

plot(liste_erreurs_train ~liste_voisins, type = "l")

# le minimum est atteint en k = 2


# Etude du compromis biais-variance
n = 50
sig = 0.5
X <- runif(n)
epsilon <- rnorm(n, mean = 0, sd = 0.5)
Y = 1 + 5*X**2 + epsilon

plot(X, Y)

x0<-0.5
Ey0<-1+5*x0^2  # Valeur en x0 de la fct de régression
Kmax<-40  # Valeur max de k
N<-10000 # nombre d'ensembles d'apprentissage
yhat<-matrix(0,N,Kmax)
y0<-rep(0,N)
for(i in 1:N){
  x<-runif(n)
  y<-1+5*x^2+sig*rnorm(n)
  d<-abs(x-x0)
  ds<-sort(d,index.return=TRUE) # tri des distances à x0
  y0[i]<-Ey0+sig*rnorm(1) # génération de Y en x0
  for(K in 1:Kmax) yhat[i,K]<-mean(y[ds$ix[1:K]]) # prédictions
}

error<-rep(0,K)
biais2<-rep(0,K)
variance<-rep(0,K)
for(K in 1:Kmax){
  error[K]<-mean((yhat[,K]-y0)^2)   # MSE
  biais2[K]<-(mean(yhat[,K])-Ey0)^2 # biais^2
  variance[K]<-var(yhat[,K])        # variance
}

plot(1:Kmax,error,type="l",
     ylim=range(error,biais2,variance),
     xlab="k", ylab="MSE",lwd=2,col="blue")
lines(1:Kmax,biais2,lty=2,lwd=2)
lines(1:Kmax,variance,lty=3,lwd=2)
lines(1:Kmax,biais2+variance+sig^2,col="red",lwd=2)