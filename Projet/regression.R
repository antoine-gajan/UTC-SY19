# Analyse de a24_reg_app.txt

library(glmnet)
library(mgcv)
library(caret)
library(class)
library(randomForest)
library(rpart)

# Chargement du jeu de données
setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/Projet")
data <- read.csv("a24_reg_app.txt", sep = " ")

set.seed(20241108)

# Caractéristiques générales du jeu de données
n = nrow(data)
p = ncol(data) - 1

# Observation de la distribution de certaines variables
boxplot(data$X1, data$X2, data$X3,
       names = c("X1", "X2", "X3"),
       main = "Boxplot of X1, X2, and X3", # Title
       ylab = "Values", # Y-axis label
       col = c("lightblue", "lightgreen", "lightcoral"))

# Analyse de la variable y
hist(data$y, xlab = "Valeurs de y", ylab = "Fréquence", main = "Histogramme de y")
boxplot(data$y, ylab = "Valeurs de y", main = "Diagramme en boite de y")

# Observation des corrélations entre 1 variable et la variable d'intérêt y
#install.packages("corrplot")
library("corrplot")

cor_data = cor(data)
cor_long <- as.data.frame(as.table(cor_data))
cor_long <- cor_long[cor_long$Var1 != cor_long$Var2, ]
cor_long <- cor_long[as.numeric(as.factor(cor_long$Var1)) < as.numeric(as.factor(cor_long$Var2)), ]
cor_long$abs_value <- abs(cor_long$Freq)
cor_long <- cor_long[order(-cor_long$abs_value), ]
top_10_correlations <- head(cor_long, 10)
print(top_10_correlations)



# Régression linéaire simple

# Séparation en jeu d'entrainement et de test
train <- sample(1:n, round(4*n/5))
data.train <- data[train, ]
data.test <- data[-train, ]

# Entrainement du modèle sur le jeu d'entrainement (entrainement direct)
lin_reg <- lm(y ~ ., data = data.train)
summary(lin_reg)

lin_reg.pred <- predict(lin_reg, data.test)
plot(lin_reg.pred ~ data.test$y, xlab = "Vraies valeurs", ylab = "Valeurs prédites", main = "Valeurs prédites vs Vraies valeurs")
abline(a = 0, b = 1)
RMSE <- sqrt(mean((lin_reg.pred - data.test$y)**2)) # RMSE = 12.70


# Avec le AIC et BIC
library(MASS)
library(leaps)
set.seed(1)

K <- 10
fold <- sample(1:K, n, replace = TRUE)
table(fold)

rms.aic <- rep(0, K)
rms.bic <- rep(0, K)

for (k in 1:K){
  cat(sprintf("Processing fold %i\n", k))
  fit <- lm(y ~ ., data = data, subset = fold != k)
  fit.aic <- stepAIC(fit)
  fit.bic <- stepAIC(fit, k = log(n))
  pred.aic <- predict(fit.aic, newdata = data[fold == k, ])
  pred.bic <- predict(fit.bic, newdata = data[fold == k, ])
  rms.aic[k] <- sqrt(mean((pred.aic - data[fold == k, ]$y)**2))
  rms.bic[k] <- sqrt(mean((pred.bic - data[fold == k, ]$y)**2))
}

print(c(mean(rms.aic), sd(rms.aic) / sqrt(K)))
print(c(mean(rms.bic), sd(rms.bic) / sqrt(K)))

# Affichage de la formule avec le BIC
print(formula(fit.bic))


# Subset selection

library('leaps')
reg.fit<-regsubsets(y~.,data=data,method='forward',nvmax=100)
plot(reg.fit,scale="r2")
reg.sum <- summary(reg.fit)
which.max(reg.sum$adjr2) # Renvoie 66 (nombre idéal de variables pour le adj r²)

# Avec Régularisation
library(glmnet)


# Ridge
x<-model.matrix(y~.,data)
y<-data$y

train<-sample(1:n,round(2*n/3))
xtrain<-x[train,]
ytrain<-y[train]
xtst<-x[-train,]
ytst<-y[-train]
cv.out<-cv.glmnet(xtrain,ytrain,alpha=0)
plot(cv.out)
fit<-glmnet(xtrain,ytrain,lambda=cv.out$lambda.min,alpha=0)
ridge.pred<-predict(fit,s=cv.out$lambda.min,newx=xtst)
print(sqrt(mean((ytst-ridge.pred)**2))) #13.23

# Lasso

cv.out<-cv.glmnet(xtrain,ytrain,alpha=1)
plot(cv.out)
fit.lasso<-glmnet(xtrain,ytrain,lambda=cv.out$lambda.min,alpha=1)
lasso.pred<-predict(fit.lasso,s=cv.out$lambda.min,newx=xtst)
print(sqrt(mean((ytst-lasso.pred)**2))) # 12.50

# ACP 
data_standardized <- scale(data[, !names(data) %in% 'y'])
acp_result <- prcomp(data_standardized, center = TRUE, scale. = TRUE)
summary(acp_result)

# Visualisation des composantes principales (biplot)
biplot(acp_result, scale = 1)

# Tracé de la variance expliquée par chaque composante principale
screeplot(acp_result, type = "lines", main = "Scree Plot des 10 premiers axes", npcs = 10)

# Diagramme des variances expliquées cumulées
explained_variance <- cumsum(acp_result$sdev^2 / sum(acp_result$sdev^2)) * 100
plot(1:length(explained_variance), explained_variance, col = "blue", type = "l", 
     xlab = "Nombre de composantes principales", ylab = "Variance expliquée cumulée (%)",
     main = "Variance expliquée cumulée par les composantes principales")


# Apprentissage sur ces données PCA
n <- nrow(data)  
p <- ncol(data) - 1

train <- sample(1:n, round(4 * n / 5))
test <- setdiff(1:n, train)

acp_results <- acp_result$x[, 1:p]
data_pca <- data.frame(acp_results, y = data$y)

train_data_pca <- data_pca[train, ]
test_data_pca <- data_pca[test, ]

K <- 5  
fold <- sample(1:K, length(train), replace = TRUE)
table(fold)

mean_rms <- rep(0, p)

for (nb_axes in 1:p) {
  rms <- rep(0, K)
  for (k in 1:K) {
    lm_pca <- lm(y ~ ., data = train_data_pca[fold != k, c(1:nb_axes, ncol(train_data_pca))])
    pred <- predict(lm_pca, newdata = train_data_pca[fold == k, c(1:nb_axes, ncol(train_data_pca))])
    rms[k] <- sqrt(mean((train_data_pca$y[fold == k] - pred)^2))
  }
  mean_rms[nb_axes] <- mean(rms)
}

# Nombre optimal de composantes principales
best_nb_axes <- which.min(mean_rms)
cat(sprintf("Le nombre optimal de composantes principales est : %i\n", best_nb_axes))

# Afficher la courbe des erreurs RMS moyennes en fonction du nombre de composantes
plot(1:p, mean_rms, type = "l", col = "blue", 
     xlab = "Nombre de composantes principales", ylab = "Erreur RMSE moyenne",
     main = "Erreur RMSE moyenne en fonction du nombre de composantes")
abline(v = best_nb_axes, col = "red", lty = 2)



# Création d'un modèle à partir des variables significativement non nulles
fit <- lm(y ~ ., data = data.train)
summary_fit <- summary(fit)
p_values <- summary_fit$coefficients[, 4] 

significant_vars <- names(p_values[p_values < 0.05 & names(p_values) != "(Intercept)"])
formula_significant <- as.formula(paste("y ~", paste(significant_vars, collapse = " + ")))

fit_significant <- lm(formula_significant, data = data)
summary(fit_significant)

pred <- predict(fit, data.test)
RMSE <- sqrt(mean((pred - data.test$y)**2)) # RMSE = 9.56

# Estimation du RMSE par validation croisée
K <- 10
fold <- sample(1:K, nrow(data.train), replace = TRUE)
table(fold)
rms.train <- rep(0, K)
rms.val <- rep(0, K)

for (k in 1:K) {
  fit <- lm(formula_significant, data = data.train, subset = fold != k)
  pred <- predict(fit, newdata = data.train[fold == k, ])
  train_pred <- predict(fit, newdata = data.train[fold != k, ])
  rms.train[k] <- sqrt(mean((train_pred - data.train[fold != k, ]$y) ^ 2))
  rms.val[k] <- sqrt(mean((pred - data.train[fold == k, ]$y) ^ 2))
}

mean(rms.val) #11.30
sd(rms.val) #1.33

# Application sur le jeu de test
reg_lin_sign <- lm(formula_significant, data = data.train)
reg_lin_sign.pred <- predict(reg_lin_sign, newdata = data.test)
RMSE <- sqrt(mean((pred - data.test$y)**2))
RMSE

plot(reg_lin_sign.pred ~ data.test$y, xlab = "Vraies valeurs", ylab = "Valeurs prédites", main = "Valeurs prédites vs Vraies valeurs")
abline(a = 0, b = 1)

# GAM
library(gam)

gam_model <- gam(y ~ .,data=data.train)
gam_model.pred <- predict(gam_model, data.test)
RMSE <- sqrt(mean((gam_model.pred - data.test$y)**2)) # RMSE = 11.26





# Fonction d'estimation des métriques de régression
calculate_metric <- function(y_true, y_pred, metric = "RMSE") {
  if (metric == "RMSE") {
    # RMSE (Root Mean Squared Error)
    return(sqrt(mean((y_pred - y_true)^2)))
  } else if (metric == "MAE") {
    # MAE (Mean Absolute Error)
    return(mean(abs(y_pred - y_true)))
  } else if (metric == "R2") {
    # R² (Coefficient de détermination)
    return(1 - sum((y_pred - y_true)^2) / sum((mean(y_true) - y_true)^2))
  } else {
    stop("Metric not recognized. Please choose 'RMSE', 'MAE', or 'R2'.")
  }
}

estimate_regression_metrics <- function(model, data.train, metric = "RMSE", n_components = 0, elastic_net_alpha = 0.5, k_neighbors = 15) {
  # Number of folds
  K <- 10
  set.seed(20240811)
  # Create fold assignments
  fold <- sample(1:K, nrow(data.train), replace = TRUE)
  
  # Initialize metric values
  metric.val <- rep(0, K)
  
  # Extract response variable and predictor matrix
  ytrain <- data.train$y
  xtrain <- model.matrix(y ~ . - 1, data.train)  # Exclude intercept
  
  for (k in 1:K) {
    # Split data into training and validation sets
    train_data <- data.train[fold != k, ]
    val_data <- data.train[fold == k, ]
    
    # If PCA is required, preprocess with PCA on training data only
    if (n_components != 0) {
      pca <- prcomp(train_data[, -which(names(train_data) == "y")], center = TRUE, scale. = TRUE)
      xtrain_pca <- predict(pca, train_data[, -which(names(train_data) == "y")])
      xval_pca <- predict(pca, val_data[, -which(names(val_data) == "y")])
      
      # Select the specified number of components
      xtrain_pca <- xtrain_pca[, 1:n_components]
      xval_pca <- xval_pca[, 1:n_components]
      
      # Update training and validation sets with PCA components
      train_data_pca <- data.frame(y = train_data$y, xtrain_pca)
      val_data_pca <- data.frame(y = val_data$y, xval_pca)
    } else {
      train_data_pca <- train_data
      val_data_pca <- val_data
    }
    
    # Model fitting and prediction
    if (model == "lm") {
      fit <- lm(y ~ ., data = train_data_pca)
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "ridge") {
      xtrain <- model.matrix(y ~ . - 1, train_data_pca)
      xval <- model.matrix(y ~ . - 1, val_data_pca)
      cv.out <- cv.glmnet(xtrain, train_data_pca$y, alpha = 0, standardize = TRUE)
      fit <- glmnet(xtrain, train_data_pca$y, lambda = cv.out$lambda.min, alpha = 0, standardize = TRUE)
      pred <- predict(fit, newx = xval)
    } else if (model == "lasso") {
      xtrain <- model.matrix(y ~ . - 1, train_data_pca)
      xval <- model.matrix(y ~ . - 1, val_data_pca)
      cv.out <- cv.glmnet(xtrain, train_data_pca$y, alpha = 1, standardize = TRUE)
      fit <- glmnet(xtrain, train_data_pca$y, lambda = cv.out$lambda.min, alpha = 1, standardize = TRUE)
      pred <- predict(fit, newx = xval)
    } else if (model == "gam") {
      predictor_names <- names(train_data_pca)[-which(names(train_data_pca) == "y")]
      formula_str <- paste("y ~", paste(predictor_names, collapse = " + "))
      fit <- gam(as.formula(formula_str), data = train_data_pca, family = gaussian())
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "elastic_net") {
      cv.out <- cv.glmnet(xtrain[fold != k, ], ytrain[fold != k], alpha = elastic_net_alpha, standardize = TRUE)
      fit <- glmnet(xtrain[fold != k, ], ytrain[fold != k], lambda = cv.out$lambda.min, alpha = elastic_net_alpha, standardize = TRUE)
      pred <- predict(fit, newx = xtrain[fold == k, ])
    } else if (model == "svm_linear") {
      fit <- svm(y ~ ., data = train_data_pca, type = "eps-regression", kernel = "linear")
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "svm_radial") {
      fit <- svm(y ~ ., data = train_data_pca, type = "eps-regression", kernel = "radial")
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "knn") {
      pred <- knn(train = train_data_pca[, -1], test = val_data_pca[, -1], cl = train_data_pca$y, k = k_neighbors)
      
    } else if (model == "tree") {
      # Entraîner un modèle d'arbre de décision
      fit <- rpart(y ~ ., data = train_data_pca)
      pred <- predict(fit, newdata = val_data_pca)
      
    } else if (model == "random_forest") {
      fit <- randomForest(y ~ ., data = train_data_pca)
      pred <- predict(fit, newdata = val_data_pca)
    }
    
    # Calculate specified metric for the validation fold
    metric.val[k] <- calculate_metric(val_data_pca$y, pred, metric)
  }
  
  # Return mean and standard deviation of the selected metric across folds
  c(mean(metric.val), sd(metric.val))
}

summary_metric_models <- function(data, metric = "RMSE", components_list_models = rep(NULL, 10)) {
  print("COMPARAISON DES MODELES DE REGRESSION")
  print(metric)
  models <- c("lm", "lasso", "ridge", "gam", "elastic_net", "svm_linear", "svm_radial", "knn", "tree", "random_forest")
  
  for (model in models) {
    result <- estimate_regression_metrics(model, data, metric = metric, elastic_net_alpha = 0.6)
    print(paste(model, ":", metric, "= Mean =", result[1], ", Std Dev =", result[2]))
    
  }
}

summary_metric_models(data, "RMSE")

plot_pca_components_model <- function(model, data, metric = "RMSE"){
  list_components <- 2:100
  list_avg_metric <- rep(0, 99)
  list_std_metric <- rep(0, 99)
  for (n_components in list_components){
    temp <- estimate_regression_metrics(model, data, metric, n_components = n_components)
    list_avg_metric[n_components - 1] <- temp[1]
    list_std_metric[n_components - 1] <- temp[2]
  }
  plot(2:100, list_avg_metric, type = "l", main = paste("PCA", metric, "results with", model), xlab = "Nombre de composantes", ylab = metric)
  min_metric_val <- min(list_avg_metric)
  min_index <- which.min(list_avg_metric) + 1
  abline(v = min_index, col = "red", lty = 2)
  print(sprintf("Le nombre de composantes optimal est : %d", min_index))
  return(min_index)
}

# Elastic net : choix du alpha (meilleur = 1)

alpha_values <- seq(0, 1, by = 0.1)

RMSE_means <- numeric(length(alpha_values))
RMSE_stds <- numeric(length(alpha_values))

for (i in seq_along(alpha_values)) {
  alpha <- alpha_values[i]
  RMSE <- estimate_regression_metrics("elastic_net", data, elastic_net_alpha = alpha)
  RMSE_means[i] <- RMSE[1]  
  RMSE_stds[i] <- RMSE[2]
}

plot(alpha_values, RMSE_means, type = "l", main = "Choix du alpha pour Elastic Net",
     xlab = "Alpha", ylab = "RMSE", col = "blue", lwd = 1)
lines(alpha_values, RMSE_means + RMSE_stds, col = "lightgreen", lty = 1)
lines(alpha_values, RMSE_means - RMSE_stds, col = "lightgreen", lty = 1)

min_RMSE <- min(RMSE_means)
optimal_alpha_index <- which.min(RMSE_means)
optimal_alpha <- alpha_values[optimal_alpha_index]
abline(v = optimal_alpha, col = "red", lty = 2)


# Choix du nombre de composantes ACP

components.lm <- plot_pca_components_model("lm", data, metric = "RMSE") # Best = 100
components.ridge <- plot_pca_components_model("ridge", data, metric = "RMSE") # Best = 100
components.lasso <- plot_pca_components_model("lasso", data, metric = "RMSE") # Best = 100
components.elastic_net <- plot_pca_components_model("elastic_net", data, metric = "RMSE") # Best = 100
components.gam <- plot_pca_components_model("gam", data, metric = "RMSE") # Best = 
components.svm_linear <- plot_pca_components_model("svm_linear", data, metric = "RMSE") # Best =
components.svm_radial <- plot_pca_components_model("svm_radial", data, metric = "RMSE") # Best =
components.knn <- plot_pca_components_model("knn", data, metric = "RMSE") # Best =
components.tree <- plot_pca_components_model("tree", data, metric = "RMSE") # Best =
components.random_forest <- 0 


# Comparaison finale des meilleurs modèles
estimate_regression_metrics("ridge", data, metric = "RMSE", n_components = 10)
