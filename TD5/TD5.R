# TD5 : Splines et modèles additifs généralisés

library(MASS)


# Exercice 1 

# Question 1 : Représentation graphique

boston <- Boston
nb_data_boston <- nrow(boston)
nb_pred_boston <- ncol(boston) - 1

train <- sample(1:n, round(4*n/5))
boston.train <- boston[train, ]
boston.test <- boston[-train, ]

plot(medv ~ lstat, data = boston)
plot(medv ~ poly(lstat^5), data = boston)

# Question 2 : Valeur optimale de p

p_tab <- seq(1, 10, 1)  # Tableau des degrés du polynôme
rms_tab <- rep(0, length(p_tab))  # Tableau des résultats RMS

for (p in p_tab){
  K <- 5
  fold <- sample(1:K, nrow(boston.train), replace = TRUE)  # Création des folds
  rms.val <- rep(0, K)  # Stocker les valeurs RMS pour chaque fold
  for (k in 1:K){  # Correction de la boucle
    # Modèle de régression polynomiale
    lin_reg <- lm(medv ~ poly(lstat, p), data = boston.train, subset = fold != k)
    # Prédictions sur les données du fold k
    pred <- predict(lin_reg, boston.train[fold == k, ])
    # Calcul du RMS pour le fold k
    rms.val[k] <- sqrt(mean((pred - boston.train[fold == k, ]$medv)^2))
  }
  # Calcul de la moyenne des RMS sur les K folds
  rms_tab[which(p_tab == p)] <- mean(rms.val)
}

plot(rms_tab ~ p_tab, main = "Erreur de validation en fonction du degré du polynôme", xlab = "Degré", ylab = "RMSE")
print(paste("Le degré qui minimise le RMSE est : ", p_tab[which.min(rms_tab)]))

fit <- lm(medv ~ poly(lstat, p_tab[which.min(rms_tab)]), data = boston.train)
ypred_poly <- predict(fit, newdata = boston.test, interval = "confidence")

ord <- order(boston.test$lstat)
plot(boston.test$lstat, boston.test$medv, cex = 0.5, xlab = "lstat", ylab = "medv", main = "Prédiction avec spline")
lines(boston.test$lstat[ord], ypred_poly[ord, "fit"], col = "red")
lines(boston.test$lstat[ord], ypred_poly[ord, "lwr"], col = "blue", lty = 2)
lines(boston.test$lstat[ord], ypred_poly[ord, "upr"], col = "blue", lty = 2)


# Question 3 : Splines naturels

library(splines)

df_tab <- seq(1, 10, 1)  
rms_tab <- rep(0, length(p_tab))

for (df in df_tab){
  K <- 5
  fold <- sample(1:K, nrow(boston.train), replace = TRUE)  # Création des folds
  rms.val <- rep(0, K)
  for (k in 1:K){ 
    # Modèle de régression polynomiale
    lin_reg <- lm(medv ~ ns(boston.train$lstat, df = df), data = boston.train, subset = fold != k)
    # Prédictions sur les données du fold k
    pred <- predict(lin_reg, boston.train[fold == k, ])
    # Calcul du RMS pour le fold k
    rms.val[k] <- sqrt(mean((pred - boston.train[fold == k, ]$medv)^2))
  }
  # Calcul de la moyenne des RMS sur les K folds
  rms_tab[which(df_tab == df)] <- mean(rms.val)
}

plot(rms_tab ~ df_tab, main = "Erreur de validation en fonction du degré du polynôme", xlab = "Degré", ylab = "RMSE")
print(paste("Le degré qui minimise le RMSE est : ", df_tab[which.min(rms_tab)]))

fit <- lm(medv ~ bs(lstat, df = 1), data = boston.train)
ypred_ns <- predict(fit, newdata = boston.test, interval = "confidence")

ord <- order(boston.test$lstat)
plot(boston.test$lstat, boston.test$medv, cex = 0.5, xlab = "lstat", ylab = "medv", main = "Prédiction avec spline")
lines(boston.test$lstat[ord], ypred_ns[ord, "fit"], col = "red")
lines(boston.test$lstat[ord], ypred_ns[ord, "lwr"], col = "blue", lty = 2)
lines(boston.test$lstat[ord], ypred_ns[ord, "upr"], col = "blue", lty = 2)

# Question 4 : Smoothing splines

ss <- smooth.spline(x = boston.train$lstat, y = boston.train$medv)
print(paste("Degré de liberté : ", ss$df))
plot(boston.train$lstat, boston.train$medv)
lines(ss$x, ss$y, col = "blue", lwd = 2)


# Question 5: Fonctions de regression sur le même graphique

plot(medv ~ lstat, data = boston, main = "Regression functions")
lines(boston.test$lstat[ord], ypred_poly[ord, "fit"], col = "red")
lines(boston.test$lstat[ord], ypred_ns[ord, "fit"], col = "green")
lines(ss$x, ss$y, col = "blue")

legend("topright", legend = c("Polynomial", "Natural splines", "Smooth splines"), col = c("blue", "green", "red"), lty = 1)

# Exercice 2 :

