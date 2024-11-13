# Chargement du jeu de données
setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/Projet")
data <- read.csv("a24_reg_app.txt", sep = " ")

# Caractéristiques générales du jeu de données
n <- nrow(data)
p <- ncol(data) - 1

# Visualisation de la distribution des variables
boxplot(data$X1, data$X2, data$X3, names = c("X1", "X2", "X3"), main = "Boxplot of X1, X2, and X3", ylab = "Values", col = c("lightblue", "lightgreen", "lightcoral"))
hist(data$y, xlab = "Valeurs de y", ylab = "Fréquence", main = "Histogramme de y")
boxplot(data$y, ylab = "Valeurs de y", main = "Diagramme en boite de y")

# Corrélations entre variables
library("corrplot")
cor_data <- cor(data)
corrplot(cor_data, method = "color", title = "Corrélations entre variables")

# Régression linéaire simple
train <- sample(1:n, round(4 * n / 5))
data.train <- data[train, ]
data.test <- data[-train, ]
lin_reg <- lm(y ~ ., data = data.train)
summary(lin_reg)
lin_reg.pred <- predict(lin_reg, data.test)
rmse <- sqrt(mean((lin_reg.pred - data.test$y) ^ 2))

# Validation croisée avec RMSE
set.seed(1)
K <- 10
fold <- sample(1:K, nrow(data.train), replace = TRUE)
rms.val <- sapply(1:K, function(k) {
  fit <- lm(y ~ ., data = data.train, subset = fold != k)
  pred <- predict(fit, newdata = data.train[fold == k, ])
  sqrt(mean((pred - data.train[fold == k, ]$y) ^ 2))
})
mean(rms.val)
sd(rms.val)

# Régression avec sélection par AIC/BIC
library(MASS)
fit.aic <- stepAIC(lin_reg)
fit.bic <- stepAIC(lin_reg, k = log(n))
summary(fit.bic)

# Sélection de sous-ensemble de variables
library(leaps)
reg.fit <- regsubsets(y ~ ., data = data, method = "forward", nvmax = 100)
summary(reg.fit)

# Ridge et Lasso
library(glmnet)
x <- model.matrix(y ~ ., data)
y <- data$y
cv.ridge <- cv.glmnet(x[train, ], y[train], alpha = 0)
cv.lasso <- cv.glmnet(x[train, ], y[train], alpha = 1)
ridge.pred <- predict(cv.ridge, newx = x[-train, ])
lasso.pred <- predict(cv.lasso, newx = x[-train, ])
ridge_rmse <- sqrt(mean((y[-train] - ridge.pred) ^ 2))
lasso_rmse <- sqrt(mean((y[-train] - lasso.pred) ^ 2))

# GAM
library(gam)
gam_model <- gam(y ~ ., data = data.train)
gam_pred <- predict(gam_model, newdata = data.test)
gam_rmse <- sqrt(mean((gam_pred - data.test$y) ^ 2))

error[K]<-mean((yhat[,K]-y0)^2)   # MSE
biais2[K]<-(mean(yhat[,K])-Ey0)^2 # biais^2
variance[K]<-var(yhat[,K])        # variance


# k-NN
library(class)
K <- 10
folds <- sample(1:K, nrow(data.train), replace = TRUE)
accuracy.train <- rep(0, 100)
accuracy.val <- rep(0, 100)

for (k in 1:100) {
  for (fold in 1:K) {
    x_train_data <- scale(data.train[folds != fold, 1:50])
    y_train_data <- data.train[folds != fold, ]$y
    x_validation_data <- scale(data.train[folds == fold, 1:50])
    y_validation_data <- data.train[folds == fold, ]$y
    
    classifier_knn <- knn(x_train_data, x_validation_data, cl = y_train_data, k = k)
    accuracy.val[fold] <- sum(classifier_knn == y_validation_data) / length(y_validation_data)
  }
  accuracy.val[k] <- mean(accuracy.val)
}

best_k <- which.max(accuracy.val)
classifier_knn <- knn(data.train[, 1:50], data.test[, 1:50], cl = data.train$y, k = best_k)
knn.accuracy.test <- sum(classifier_knn == data.test$y) / length(data.test$y)

# Régression logistique
library(nnet)
reg_log <- multinom(y ~ ., data = data.train)
reg_log.accuracy.test <- sum(predict(reg_log, newdata = data.test[, 1:50]) == data.test$y) / length(data.test$y)

# QDA
library(MASS)
qda <- qda(y ~ ., data = data.train)
qda.accuracy.test <- sum(predict(qda, newdata = data.test)$class == data.test$y) / length(data.test$y)

# SVM radial
library(e1071)
svm_model <- svm(y ~ ., data = scale(data.train), type = "C-classification", kernel = "radial", cross = 10)
svm.accuracy.test <- sum(predict(svm_model, newdata = scale(data.test)) == data.test$y) / length(data.test$y)

# Random Forest
library(randomForest)
rf_model <- randomForest(as.factor(y) ~ ., data = scale(data.train))
rf.accuracy.test <- sum(predict(rf_model, newdata = scale(data.test)) == data.test$y) / length(data.test$y)

# ACP
acp_result <- prcomp(scale(data[, -p]), center = TRUE)
screeplot(acp_result, type = "lines")
explained_variance <- cumsum(acp_result$sdev^2 / sum(acp_result$sdev^2)) * 100
