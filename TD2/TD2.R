# 1. Données prostate

# Chargement des données de prostate
prostate <- read.csv("prostate.data", sep = "\t")
prostate <- subset(prostate, select = -X)


# Régression linéaire

fit <- lm(lpsa ~ ., data = prostate)
summary(fit)

# Calcul des intervalles de confiance sur les coefficients
confint(fit)

# Affichage des yi prédits en fonction des yi
plot(fit$fitted.values ~ prostate$lpsa, xlab = "Valeurs y", ylab = "Valeurs y prédites", main = "Valeurs prédites en fonction \n des valeurs réelles")

# Apprentissage avec jeu d'apprentissage
train_data <- prostate[prostate$train, c("lcavol", "lweight", "age", "lbph", "lpsa")]
test_data <- prostate[!prostate$train, c("lcavol", "lweight", "age", "lbph", "lpsa")]

fit <- lm(lpsa ~ ., data = train_data)
p <- predict(fit, test_data, interval = 'confidence')

# Erreur quadratique moyenne
mse <- mean((test_data$lpsa - p)**2) # Renvoie 0.50

# Affichage intervalles de confiance
plot(test_data$lpsa, p[, "fit"], main = "Valeurs prédites\navec intervalles de confiance", 
     xlab = "Valeurs réelles", ylab = "Valeurs prédites", pch = 20, col = "blue")
arrows(test_data$lpsa, p[, "lwr"], test_data$lpsa, p[, "upr"], 
       length = 0.05, angle = 90, code = 3, col = "red")

# 2. Données Boston

library(MASS)

# Exploration graphique des données
Boston <- Boston[, c("medv", "crim", "nox", "dis","lstat")]
plot(Boston)

plot(Boston$medv ~ Boston$crim)
plot(Boston$medv ~ Boston$nox)
plot(Boston$medv ~ Boston$dis)
plot(Boston$medv ~ Boston$lstat)

# Partitionner data
set.seed(101)
sample <- sample.int(n = nrow(Boston), size = floor(.75*nrow(Boston)), replace = F)
train <- Boston[sample, ]
test  <- Boston[-sample, ]

# Regression linéraire sur les données d'apprentissage

fit <- lm(medv ~., data = train)
summary(fit)

# Prédiction sur les données de test

p <- predict(fit, test, interval = "confidence")
# Erreur quadratique moyenne
mse <- mean((test$medv - p)**2) # Renvoie 36.61

# Affichage intervalles de confiance
plot(test$medv, p[, "fit"], main = "Valeurs prédites\navec intervalles de confiance", 
     xlab = "Valeurs réelles", ylab = "Valeurs prédites", pch = 20, col = "blue")
arrows(test$medv, p[, "lwr"], test$medv, p[, "upr"], 
       length = 0.05, angle = 90, code = 3, col = "red")
abline(a=0, b=1)

# Ajout des variables au carré
Boston_square <- transform(Boston, crim_2 = crim^2, nox_2 = nox^2, dis_2 = dis^2, lstat_2 = lstat^2)
# Partitionner data
set.seed(101)
sample <- sample.int(n = nrow(Boston), size = floor(.75*nrow(Boston)), replace = F)
train <- Boston_square[sample, ]
test  <- Boston_square[-sample, ]


fit <- lm(medv ~., data = train)
summary(fit)

# Prédiction sur les données de test

p <- predict(fit, test, interval = "confidence")
# Erreur quadratique moyenne
mse <- mean((test$medv - p)**2) # Renvoie 26.76

# Affichage intervalles de confiance
plot(test$medv, p[, "fit"], main = "Valeurs prédites\navec intervalles de confiance", 
     xlab = "Valeurs réelles", ylab = "Valeurs prédites", pch = 20, col = "blue")
arrows(test$medv, p[, "lwr"], test$medv, p[, "upr"], 
       length = 0.05, angle = 90, code = 3, col = "red")
abline(a=0, b=1)


# 3. Intervalles de confiance et de prédiction

beta0_ok <- 0
beta1_ok <- 0
beta2_ok <- 0
all_beta_ok <- 0
pred_ok <- 0
N <- 1000
# 1000 itérations de génération de jeux de données
for (app in 1:N) {
  # Génération d'un jeu avec 100 données
  X1 = runif(100)
  X2 = runif(100)
  sig <- 1
  epsilon <- rnorm(10000, 0, sig)
  
  coeff = sample(-10:10, 3, replace = TRUE)
  Y = coeff[1] + coeff[2] * X1 + coeff[3] * X2 + epsilon
  
  data <- data.frame(X1, X2, Y)
  
  # Apprentissage du modèle
  fit <- lm(Y ~ ., data = data)
  
  # Calcul des intervalles de confiance sur les coefficients
  beta_hat <- confint(fit)
  
  check0 <- (coeff[1] >= beta_hat[1, 1] && coeff[1] <= beta_hat[1, 2])
  check1 <- (coeff[2] >= beta_hat[2, 1] && coeff[2] <= beta_hat[2, 2])
  check2 <- (coeff[3] >= beta_hat[3, 1] && coeff[3] <= beta_hat[3, 2])
  
  beta0_ok <- beta0_ok + check0
  beta1_ok <- beta1_ok + check1
  beta2_ok <- beta2_ok + check2
  
  if (check0 && check1 && check2){
    all_beta_ok <- all_beta_ok + 1
  }
  
  # Vérification pour une nouvelle donnée
  x0 = sample(-10:10, 2, replace = TRUE)
  Y0 = coeff[1] + coeff[2] * x0[1] + coeff[3] * x0[2] + epsilon
  x0 <- data.frame(X1 = x0[1], X2 = x0[2])
  p <- predict(fit, x0, interval = "prediction")
  check_pred <- (Y0 >= p[1, "lwr"] && Y0 <= p[1, "upr"])
  pred_ok <- pred_ok + check_pred
}

sprintf("Taux beta0 : %.2f%%", beta0_ok / N * 100)
sprintf("Taux beta1 : %.2f%%", beta1_ok / N * 100)
sprintf("Taux beta2 : %.2f%%", beta2_ok / N * 100)
sprintf("Tous les beta dans l'intervalle de confiance : %.2f%%", all_beta_ok / N * 100)
sprintf("Taux prédiction : %.2f%%", pred_ok / N * 100)


