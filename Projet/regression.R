# Analyse de a24_reg_app.txt

library(glmnet)
library(mgcv)
library(caret)
library(class)
library(randomForest)
library(rpart)
library(e1071)

# Chargement du jeu de données
setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/Projet")
data <- read.csv("a24_reg_app.txt", sep = " ")

set.seed(20241108)
set.seed(123)

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
par(mfrow=c(1, 3))
hist(data$y, xlab = "Valeurs de y", ylab = "Fréquence", main = "Histogramme de y")
hist(data$X8, xlab = "Valeurs de X8", ylab = "Fréquence", main = "Histogramme de X8")
hist(data$X75, xlab = "Valeurs de X77", ylab = "Fréquence", main = "Histogramme de X75")

# Test de la normalité de y
resultat <- shapiro.test(data$y)
ks.result <- ks.test(data$X8, "punif", min = 0, max = 10)

par(mfrow=c(1, 1))

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


preprocess_data_frame <- function(df, rescale = TRUE) {
  target_var <- "y"
  
  # Separate predictors and target variable
  predictors <- df[, !(names(df) %in% target_var)]  # Exclude 'y'
  target <- df[[target_var]]
  
  # Rescale the data
  if (rescale == TRUE) {
    numeric_columns <- sapply(predictors, is.numeric)
    predictors[, numeric_columns] <- scale(predictors[, numeric_columns])
  }
  
  # One-hot encode all factor variables in predictors
  encoded_predictors <- model.matrix(~ . + 0, data = predictors)
  
  # Combine the encoded predictors with the target variable
  final_data <- data.frame(encoded_predictors, y = target)
  return(final_data)
}

data <- preprocess_data_frame(data)

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
library(formula.tools)


K <- 10
n <- nrow(data)  # Assurez-vous que `data` est défini avant d'exécuter ce code
fold <- sample(1:K, n, replace = TRUE)
table(fold)

# Initialisation des vecteurs pour stocker les résultats
rms.bic <- numeric(K)
formulas <- character(K)

for (k in 1:K) {
  cat(sprintf("Processing fold %i\n", k))
  
  # Séparer les données d'entraînement et de test
  train <- data[fold != k, ]
  test <- data[fold == k, ]
  
  # Standardisation des données d'entraînement
  train_scaled <- scale(model.matrix(y ~ . - 1, train))
  reg.scaling_params_center <- attr(train_scaled, "scaled:center")
  reg.scaling_params_sd <- attr(train_scaled, "scaled:scale")
  
  # Standardisation des données de test avec les mêmes paramètres
  test_scaled <- scale(model.matrix(y ~ . - 1, test),
                       center = reg.scaling_params_center,
                       scale = reg.scaling_params_sd)
  
  # Ajouter la variable cible (y) aux données standardisées
  train_scaled <- data.frame(train_scaled, y = train$y)
  test_scaled <- data.frame(test_scaled, y = test$y)
  
  # Ajustement du modèle initial et sélection selon BIC
  fit <- lm(y ~ ., data = train_scaled)
  fit.bic <- stepAIC(fit, k = 3, direction = "backward", trace = FALSE)
  
  # Prédiction et calcul du RMSE
  pred.bic <- predict(fit.bic, newdata = test_scaled)
  rms.bic[k] <- sqrt(mean((pred.bic - test_scaled$y)^2))
  
  # Stockage de la formule sélectionnée
  formulas[k] <- paste(deparse(formula(fit.bic)), collapse = " ")
}

# Cross-validation pour évaluer chaque formule
cv_rmse <- numeric(K)

for (k in 1:K) {
  cat(sprintf("Cross-validating formula %i\n", k))
  current_formula <- formulas[k]
  
  # Créer de nouvelles partitions pour la validation croisée interne
  inner_fold <- sample(1:K, n, replace = TRUE)
  inner_rmse <- numeric(K)
  
  for (j in 1:K) {
    # Diviser les données selon les folds internes
    train_inner <- data[inner_fold != j, ]
    test_inner <- data[inner_fold == j, ]
    
    # Standardisation des données d'entraînement
    train_scaled <- scale(model.matrix(y ~ . - 1, train_inner))
    reg.scaling_params_center <- attr(train_scaled, "scaled:center")
    reg.scaling_params_sd <- attr(train_scaled, "scaled:scale")
    
    # Standardisation des données de test avec les mêmes paramètres
    test_scaled <- scale(model.matrix(y ~ . - 1, test_inner),
                         center = reg.scaling_params_center,
                         scale = reg.scaling_params_sd)
    
    # Ajouter la variable cible (y) aux données standardisées
    train_scaled <- data.frame(train_scaled, y = train_inner$y)
    test_scaled <- data.frame(test_scaled, y = test_inner$y)
    
    # Ajuster le modèle avec la formule actuelle
    fit <- lm(as.formula(current_formula), data = train_scaled)
    pred <- predict(fit, newdata = test_scaled)
    
    # Calcul du RMSE pour le fold interne
    inner_rmse[j] <- sqrt(mean((pred - test_inner$y)^2))
  }
  
  # Stocker la moyenne des RMSE internes
  cv_rmse[k] <- mean(inner_rmse)
}

# Afficher les RMSE validés par cross-validation
print(cv_rmse)



# Importance des variables

rf_model <- randomForest(x = data[, -which(names(data) == "y")], y = data$y, 
                         ntree = 500, 
                         mtry = (ncol(data)) / 2,
                         nodesize = 1,
                         importance = TRUE, 
                         keep.forest = TRUE)

importance_scores <- importance(rf_model, type = 1)  # type = 1 gives Mean Decrease in Accuracy



varImpPlot(rf_model)


# ACP 
data_standardized <- scale(data[, !names(data) %in% 'y'])
acp_result <- prcomp(data_standardized, center = TRUE, scale. = TRUE)
summary(acp_result)

# Visualisation des composantes principales (biplot)
biplot(acp_result, scale = 1)

# Tracé de la variance expliquée par chaque composante principale
screeplot(acp_result, type = "lines", main = "Scree Plot des 10 premiers axes", npcs = 10)

library(gridExtra)
library(ggplot2)

# Diagramme des variances expliquées cumulées
explained_variance <- cumsum(acp_result$sdev^2 / sum(acp_result$sdev^2)) * 100

p1 <- ggplot(data.frame(x = 1:length(explained_variance), y = explained_variance),
       aes(x = x, y = y)) +
  geom_line(color = "blue") +
  labs(title = "Variance expliquée cumulée par les composantes principales",
       x = "Nombre de composantes principales",
       y = "Variance expliquée cumulée (%)") +
  theme_minimal()

pc_data <- acp_result$x[, 1:2]  # Les deux premières composantes principales
y <- data$y  # Variable cible


p2 <- ggplot(pc_data, aes(x = PC1, y = PC2, color = y)) +
  geom_point(size = 2) +
  scale_color_gradient(low = "blue", high = "red") +  # Color gradient from blue to red
  labs(title = "Représentation des données en 2 dimensions",
       x = "Première composante principale (PC1)",
       y = "Deuxième composante principale (PC2)",
       color = "Valeur de y") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)


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

estimate_regression_metrics <- function(model, data.train, metric = "RMSE", n_components = 0, elastic_net_alpha = 0.5, k_neighbors = 28, formula_str = "y ~ .") {
  set.seed(20241108)
  # Number of folds
  K <- 10
  # Create fold assignments
  fold <- sample(1:K, nrow(data.train), replace = TRUE)
  
  # Initialize metric values
  metric.val <- rep(0, K)
  
  # Extract response variable and predictor matrix
  ytrain <- data.train$y
  # Create formula from string for dynamic formula
  formula <- as.formula(formula_str)
  
  for (k in 1:K) {
    # Split data into training and validation sets
    train_data <- data.train[fold != k, ]
    val_data <- data.train[fold == k, ]
    
    predictors <- setdiff(names(data.train), "y")
    
    # Compute normalization parameters from training data
    scaling_params <- apply(train_data[, predictors, drop = FALSE], 2, function(x) {
      c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE))
    })
    
    # Normalize training data
    train_data[, predictors] <- scale(
      train_data[, predictors, drop = FALSE],
      center = scaling_params["mean", ],
      scale = scaling_params["sd", ]
    )
    
    # Normalize validation data using training parameters
    val_data[, predictors] <- scale(
      val_data[, predictors, drop = FALSE],
      center = scaling_params["mean", ],
      scale = scaling_params["sd", ]
    )
    
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
      fit <- lm(formula, data = train_data_pca)
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "ridge") {
      xtrain <- model.matrix(formula, train_data_pca)
      xval <- model.matrix(formula, val_data_pca)
      cv.out <- cv.glmnet(xtrain, train_data_pca$y, alpha = 0, standardize = TRUE)
      fit <- glmnet(xtrain, train_data_pca$y, lambda = cv.out$lambda.min, alpha = 0, standardize = TRUE)
      pred <- predict(fit, newx = xval)
    } else if (model == "lasso") {
      xtrain <- model.matrix(formula, train_data_pca)
      xval <- model.matrix(formula, val_data_pca)
      cv.out <- cv.glmnet(xtrain, train_data_pca$y, alpha = 1, standardize = TRUE)
      fit <- glmnet(xtrain, train_data_pca$y, lambda = cv.out$lambda.min, alpha = 1, standardize = TRUE)
      pred <- predict(fit, newx = xval)
    } else if (model == "gam") {
      fit <- gam(as.formula(paste("y ~", paste("s(", all.vars(formula)[-1], ")", collapse = " + "))), data = train_data_pca, family = gaussian())
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "elastic_net") {
      xtrain <- model.matrix(formula, train_data_pca)
      xval <- model.matrix(formula, val_data_pca)
      cv.out <- cv.glmnet(xtrain, train_data_pca$y, alpha = elastic_net_alpha, standardize = TRUE)
      fit <- glmnet(xtrain, train_data_pca$y, lambda = cv.out$lambda.min, alpha = elastic_net_alpha, standardize = TRUE)
      pred <- predict(fit, newx = xval)
    } else if (model == "svm_linear") {
      fit <- svm(formula, data = train_data_pca, type = "eps-regression", kernel = "linear")
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "svm_radial") {
      fit <- svm(formula, data = train_data_pca, type = "eps-regression", kernel = "radial")
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "svm_poly") {
      fit <- svm(formula, data = train_data_pca, type = "eps-regression", kernel = "polynomial")
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "knn") {
      fit <- knnreg(x = train_data_pca[, -which(names(train_data) == "y")], y = train_data_pca$y, k = k_neighbors)
      pred <- predict(fit, newdata = val_data_pca[, -which(names(train_data) == "y")])
    } else if (model == "tree") {
      fit <- rpart(formula, data = train_data_pca)
      pred <- predict(fit, newdata = val_data_pca)
    } else if (model == "random_forest") {
      fit <- randomForest(formula, data = train_data_pca)
      pred <- predict(fit, newdata = val_data_pca)
    } 
    
    # Calculate specified metric for the validation fold
    metric.val[k] <- calculate_metric(val_data_pca$y, pred, metric)
  }
  
  # Return mean and standard deviation of the selected metric across folds
  c(mean(metric.val), sd(metric.val))
}

estimate_regression_metrics_2 <- function(data.train, metric = "RMSE", n_components = 0, elastic_net_alpha = 0.5, k_neighbors = 28, formula_str = "y ~ .") {
  set.seed(20241108)
  # Number of folds
  K <- 10
  # Create fold assignments
  fold <- sample(1:K, nrow(data.train), replace = TRUE)
  
  # Initialize metric values
  metric.val <- rep(0, K)
  
  # Extract response variable and predictor matrix
  ytrain <- data.train$y
  # Create formula from string for dynamic formula
  formula <- as.formula(formula_str)
  
  for (k in 1:K) {
    # Split data into training and validation sets
    train_data <- data.train[fold != k, ]
    val_data <- data.train[fold == k, ]
    
    predictors <- setdiff(names(data.train), "y")
    
    # Compute normalization parameters from training data
    scaling_params <- apply(train_data[, predictors, drop = FALSE], 2, function(x) {
      c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE))
    })
    
    # Normalize training data
    train_data[, predictors] <- scale(
      train_data[, predictors, drop = FALSE],
      center = scaling_params["mean", ],
      scale = scaling_params["sd", ]
    )
    
    # Normalize validation data using training parameters
    val_data[, predictors] <- scale(
      val_data[, predictors, drop = FALSE],
      center = scaling_params["mean", ],
      scale = scaling_params["sd", ]
    )
    
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
      fit <- lm(formula, data = train_data_pca)
      pred <- predict(fit, newdata = val_data_pca)
      
      xtrain <- model.matrix(formula, train_data_pca)
      xval <- model.matrix(formula, val_data_pca)
      cv.out <- cv.glmnet(xtrain, train_data_pca$y, alpha = elastic_net_alpha, standardize = TRUE)
      fit <- glmnet(xtrain, train_data_pca$y, lambda = cv.out$lambda.min, alpha = elastic_net_alpha, standardize = TRUE)
      pred <- pred + predict(fit, newx = xval)
      
      fit <- svm(formula, data = train_data_pca, type = "eps-regression", kernel = "linear")
      pred <- pred + predict(fit, newdata = val_data_pca)
      pred <- pred / 3
     
    
    # Calculate specified metric for the validation fold
    metric.val[k] <- calculate_metric(val_data_pca$y, pred, metric)
  }
  
  # Return mean and standard deviation of the selected metric across folds
  c(mean(metric.val), sd(metric.val))
}


summary_metric_models <- function(data, metric = "RMSE", components_list_models = rep(NULL, 10), formule = "y ~.") {
  print("COMPARAISON DES MODELES DE REGRESSION")
  print(metric)
  models <- c("lm", "lasso", "ridge", "elastic_net", "svm_poly", "svm_linear", "svm_radial", "knn", "tree", "random_forest")
  
  for (model in models) {
    result <- estimate_regression_metrics(model, data, metric = metric, elastic_net_alpha = 0.7, formula = formule)
    print(paste(model, ":", metric, "= Mean =", result[1], ", Std Dev =", result[2]))
    
  }
}

summary_metric_models_2 <- function(data, metric = "RMSE", components_list_models = rep(NULL, 10), formule = "y ~.") {
  print("COMPARAISON DES MODELES DE REGRESSION")
  print(metric)
  result <- estimate_regression_metrics_2(data, metric = metric, elastic_net_alpha = 0.7, formula = formule)
  print(paste(metric, "= Mean =", result[1], ", Std Dev =", result[2]))
}

formula_significant <- "y ~ X1 + X2 + X3 + X4 + X8 + X13 + X15 + X20 + X21 + X25 + X27 + X29 + X30 + X31 + X33 + X42 + X43 + X44 + X45 + X46 + X47 + X48 + X51 + X54 + X55 + X56 + X57 + X58 + X59 + X61 + X62 + X69 + X71 + X72 + X76 + X80 + X81 + X84 + X87 + X89 + X92 + X93 + X96 + X99"


summary_metric_models(data, "RMSE", formule = formula_significant)
summary_metric_models_2(data, "RMSE", formule = formula_significant)


plot_pca_components_models <- function(models, data, metric = "RMSE", max_components = 100) {
  # Prepare components to evaluate
  list_components <- 2:max_components
  
  # Set up the plot
  plot(NULL, xlim = range(list_components), ylim = c(10, 50),
       xlab = "Nombre de composantes principales",
       ylab = metric,
       main = paste("Évolution de", metric, "avec PCA"))
  
  colors <- rainbow(length(models))  # Assign colors to each model
  legend_labels <- c()
  
  # Loop through models
  for (i in seq_along(models)) {
    model_name <- models[i]
    list_avg_metric <- rep(0, length(list_components))
    
    for (n_components in list_components) {
      # Use the estimate_regression_metrics function to calculate metrics
      temp <- estimate_regression_metrics(model_name, data, metric, n_components = n_components)
      list_avg_metric[n_components - 1] <- temp[1]
    }
    
    # Plot the metric evolution for this model
    lines(list_components, list_avg_metric, col = colors[i], lwd = 1)
    legend_labels <- c(legend_labels, model_name)
    
    # Mark the optimal number of components
    min_metric_val <- min(list_avg_metric)
    min_index <- which.min(list_avg_metric) + 1
    print(sprintf("Le nombre de composantes optimal pour %s est : %d", model_name, min_index))
  }
  
  # Add legend
  legend("topright", legend = legend_labels, col = colors, lwd = 2, lty = 1)
}

models <- c("lm", "ridge", "lasso", "knn")
plot_pca_components_models(models, data, metric = "RMSE", max_components = 100)


# Elastic net : choix du alpha (meilleur = 1)
par(mfrow=c(1, 2))
alpha_values <- seq(0, 1, by = 0.1)

RMSE_means <- numeric(length(alpha_values))
RMSE_stds <- numeric(length(alpha_values))

for (i in seq_along(alpha_values)) {
  alpha <- alpha_values[i]
  RMSE <- estimate_regression_metrics("elastic_net", data, elastic_net_alpha = alpha, formula_str = formula_significant)
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

# KNN : choix du nombre de voisins

neighbors <- seq(1, 50, by = 1)

RMSE_means <- numeric(length(neighbors))
RMSE_stds <- numeric(length(neighbors))

for (i in seq_along(neighbors)) {
  voisin <- neighbors[i]
  RMSE <- estimate_regression_metrics("knn", data, k_neighbors = voisin)
  RMSE_means[i] <- RMSE[1]  
  RMSE_stds[i] <- RMSE[2]
}

plot(neighbors, RMSE_means, type = "l", main = "Choix du nombre de voisins pour KNN",
     xlab = "Nombre de voisins", ylab = "RMSE", col = "blue", lwd = 1)

min_RMSE <- min(RMSE_means)
optimal_voisin_index <- which.min(RMSE_means)
optimal_voisin <- neighbors[optimal_voisin_index]
abline(v = optimal_voisin, col = "red", lty = 2)



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


# Réseau de neurones

library(neuralnet)

fit <- neuralnet(
  formula = y ~., 
  data = data.train, 
  hidden = c(64, 32, 8), 
  linear.output = TRUE,
  stepmax = 1e6
)
pred <- compute(fit, data.test)$net.result

rmse <- calculate_metric(pred, data.test$y, "RMSE")
print(rmse)
# Summary of the Model
plot(nn_model)