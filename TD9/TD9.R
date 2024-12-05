# TD9 : SVM
library(caret)
library(gam)
library(Matrix)
library(e1071) 
library(class) 
library(nnet)
library(MASS)
library(randomForest)

## Chargement du jeu de données
setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/TD9")
load("data_expressions.RData")

## 0. Affichage d'une image
par(mfrow = c(2, 2))
for (i in 1:4){
  I<-matrix(X[i,],60,70)
  I1 <- apply(I, 1, rev)
  image(t(I1),col=gray(0:255 / 255))
}

## 1. Prétraitement des données
par(mfrow = c(1, 1))

# Enlever les variables constantes
X <- X[,apply(X, 2, var, na.rm=TRUE) != 0]
acp_result <- prcomp(X, center = TRUE, scale. = FALSE)

# Tracé de la variance expliquée par chaque composante principale
screeplot(acp_result, type = "lines", main = "Scree Plot des 10 premiers axes", npcs = 10)

# Diagramme des variances expliquées cumulées
explained_variance <- cumsum(acp_result$sdev^2 / sum(acp_result$sdev^2)) * 100
plot(1:length(explained_variance), explained_variance, col = "blue", type = "l", 
     xlab = "Nombre de composantes principales", ylab = "Variance expliquée cumulée (%)",
     main = "Variance expliquée cumulée par les composantes principales")

# Représentation en 2 dimensions
X_pca <- as.data.frame(acp_result$x[, 1:100])

# Partition de l'ensemble
train_indices <- sample(1:nrow(X_pca), size = round(0.8 * nrow(X_pca)))
X_pca.train <- X_pca[train_indices, ]
X_pca.test <- X_pca[-train_indices, ]

y_binary <- as.factor(ifelse(y %in% c("joy","surprise"), 1, 0))
y_binary.train <- y_binary[train_indices]
y_binary.test <- y_binary[-train_indices]


## 2. SVM linéaire

K <- 10
fold <- sample(1:K, length(y_binary.train), replace = TRUE)

C_values <- c(0.01, 0.1, 1, 10, 100) 
svm.accuracy <- numeric(length(C_values)) 

for (i in seq_along(C_values)) { 
  c <- C_values[i]
  svm.fold_accuracy <- numeric(K) 
  
  for (k in 1:K) {
    svm_model <- svm(
      x = X_pca.train[fold != k, ], 
      y = as.factor(y_binary.train[fold != k]), 
      cost = c, 
      kernel = "linear"
    )
    pred <- predict(svm_model, newdata = X_pca.train[fold == k, ])
    svm.fold_accuracy[k] <- mean(pred == y_binary.train[fold == k])
  }
  
  # Moyenne des précisions pour ce C
  svm.accuracy[i] <- mean(svm.fold_accuracy)
}

# Afficher les résultats
data.frame(C_values, svm.accuracy)

# Apprentissage du meilleur modèle

svm_model <- svm(
  x = X_pca.train, 
  y = as.factor(y_binary.train), 
  cost = 0.01, 
  kernel = "linear", 
  probability = TRUE
)
svm_model.pred <- predict(svm_model, newdata = X_pca.test)

svm_model.test_error <- mean(svm_model.pred != y_binary.test)
print(svm_model.test_error)

# Courbe ROC

library(pROC)

svm_model.pred_prob <- predict(svm_model, newdata = X_pca.test, probability = TRUE)
probabilities <- attr(svm_model.pred_prob, "probabilities")[, 1]
roc_curve <- roc(y_binary.test, probabilities, levels = c(0, 1))

# Tracer la courbe ROC
plot(roc_curve, col = "blue", main = "SVM ROC Curve")

# Calculer et afficher l'AUC
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

## Question 3 : Noyau non linéaire
library(kernlab)

get_results_svm <- function(X_pca.train, y_binary.train, kernel = "vanilladot") {
  K <- 10
  fold <- sample(1:K, length(y_binary.train), replace = TRUE)
  
  C_values <- c(0.01, 0.1, 1, 10, 100)
  svm.accuracy <- numeric(length(C_values))
  
  for (i in seq_along(C_values)) {
    c <- C_values[i]
    # Ajuster le modèle SVM
    svm_model <- ksvm(
      x = X_pca.train[fold != k, ],
      y = as.factor(y_binary.train[fold != k]),
      type = "C-svc",
      kernel = kernel, 
      cross = 10,
      C = c
    )
    # Prédictions
    pred <- predict(svm_model, newdata = X_pca.train[fold == k, ])
    svm.accuracy[i] <- mean(svm.fold_accuracy)
  }
  
  # Retourner les résultats
  return(data.frame(C_values, svm.accuracy))
}

# Tester les kernels disponibles dans kernlab
svm.linear <- get_results_svm(X_pca.train, y_binary.train, kernel = "vanilladot")
svm.rbf <- get_results_svm(X_pca.train, y_binary.train, kernel = "rbfdot")
svm.poly <- get_results_svm(X_pca.train, y_binary.train, kernel = "polydot")

# Question 4

data = data.frame(X_pca, y_binary)
data.train = data.frame(X_pca.train, y = y_binary.train)
data.test = data.frame(X_pca.test, y = y_binary.test)

fit <- multinom(y ~ ., data = data.train)
pred <- predict(fit, newdata = data.test, type = 'class')
mean(pred == y_binary.test)

fit <- lda(y ~ ., data = data.train)
pred <- predict(fit, newdata = data.test)$class
mean(pred == y_binary.test)

fit <- naiveBayes(y ~ ., data = data.train)
pred <- predict(fit, newdata = data.test)
mean(pred == y_binary.test)

fit <- randomForest(y ~ ., data = data.train)
pred <- predict(fit, newdata = data.test)
mean(pred == y_binary.test)
