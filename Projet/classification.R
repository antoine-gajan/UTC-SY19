# Analyse de a24_reg_app.txt

library(mclust)
library(Matrix)
library(corrplot)
library(e1071) 
library(caTools) 
library(class) 
library(nnet)
library(MASS)
library(stats)
library(rpart)
library(rpart.plot)
library(randomForest)
library(MLmetrics)


# Chargement du jeu de données
setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/Projet")
data <- read.csv("a24_clas_app.txt", sep = " ")
data$y <- factor(data$y)

set.seed(20241108)
# Caractéristiques générales du jeu de données
n = nrow(data)
p = ncol(data) - 1

X <- as.matrix(data[, 1:p])
rankMatrix(X)

cor_data = cor(data)
cor_long <- as.data.frame(as.table(cor_data))
cor_long <- cor_long[cor_long$Var1 != cor_long$Var2, ]
cor_long <- cor_long[as.numeric(as.factor(cor_long$Var1)) < as.numeric(as.factor(cor_long$Var2)), ]
cor_long$abs_value <- abs(cor_long$Freq)
cor_long <- cor_long[order(-cor_long$abs_value), ]
top_10_correlations <- head(cor_long, 10)
print(top_10_correlations)

# Séparation en jeu d'entrainement de et test
train <- sample(1:n, round(4*n/5))
data.train <- data[train, ]
data.test <- data[-train, ]

# Observation de la distribution de certaines variables
boxplot(data$X1, data$X2, data$X3,
        names = c("X1", "X2", "X3"),
        main = "Boxplot of X1, X2, and X3", # Title
        ylab = "Values", # Y-axis label
        col = c("lightblue", "lightgreen", "lightcoral"))

plot(data$X2 ~ data$X1, main = "Valeurs de X2 en fonction de X1")

barplot(table(data$y), main = "Distribution des valeurs de y", xlab = "y", ylab = "Nombre", col = "lightblue", border = "black")

resultat <- shapiro.test(data$X21)
ks.result <- ks.test(data$X8, "punif", min = 0, max = 10)



par(mfrow=c(1, 3))
hist(data$X1, main = "Variable X1", xlab = "X1")
hist(data$X21, main = "Variable X21", xlab = "X21")
barplot(table(data$X47), main = "Variable X46", xlab = "X46")


# KNN : Fonction pour trouver le meilleur nombre de voisins k
find_best_k <- function(data.train, max_k = 100, K = 10) {
  
  # Standardisation des données
  data.train.x_scale <- scale(data.train[, 1:50])
  
  # Création des folds pour la validation croisée
  folds <- sample(1:K, nrow(data.train), replace = TRUE)
  
  # Initialisation des vecteurs pour stocker l'accuracy
  accuracy.train <- rep(0, max_k)
  accuracy.val <- rep(0, max_k)
  
  # Boucle pour tester les différentes valeurs de k
  for (k in 1:max_k) {  
    accuracy.train.fold <- rep(0, K)
    accuracy.val.fold <- rep(0, K)
    
    # Pour chaque pli de la validation croisée
    for (fold in 1:K) {
      # Données d'entraînement et de validation pour le pli courant
      x_train_data <- data.train.x_scale[folds != fold, ]
      y_train_data <- data.train[folds != fold, ]$y
      x_validation_data <- data.train.x_scale[folds == fold, ]
      y_validation_data <- data.train[folds == fold, ]$y
      
      # Application du k-NN pour les données de validation
      classifier_knn_val <- knn(train = x_train_data, test = x_validation_data, cl = y_train_data, k = k)
      # Calcul de l'accuracy pour ce pli sur les données de validation
      accuracy.val.fold[fold] <- sum(classifier_knn_val == y_validation_data) / length(y_validation_data)
      
      # Application du k-NN pour les données d'entraînement
      classifier_knn_train <- knn(train = x_train_data, test = x_train_data, cl = y_train_data, k = k)
      # Calcul de l'accuracy pour ce pli sur les données d'entraînement
      accuracy.train.fold[fold] <- sum(classifier_knn_train == y_train_data) / length(y_train_data)
    }
    
    # Moyenne des accuracies sur tous les plis
    accuracy.val[k] <- mean(accuracy.val.fold, na.rm = TRUE)
    accuracy.train[k] <- mean(accuracy.train.fold, na.rm = TRUE)
  }
  
  # Détermination du k optimal (max de l'accuracy de validation)
  best_k <- which.max(accuracy.val)
  max_accuracy_val <- max(accuracy.val)
  
  # Affichage du graphique
  plot(1:max_k, accuracy.train, main = "Accuracy sur les données d'entrainement et de validation avec KNN",
       col = "blue", type = "l", xlab = "Nombre de voisins (k)", ylab = "Accuracy")
  lines(1:max_k, accuracy.val, col = "green")
  abline(v = best_k, col = "red", lty = 2)
  legend("topright", legend = c("Entraînement", "Validation"), col = c("blue", "green"), lty = 1)
  
  # Résultat
  message(sprintf("Le nombre de voisins optimal est : %d avec une accuracy de validation de %.2f%%", 
                  best_k, max_accuracy_val * 100))
  
  return(best_k)
}

best_k <- find_best_k(data)


# Régression logistique classique

reg_log <- multinom(y ~ ., data = data.train)
reg_log.pred <- predict(reg_log, newdata=data.test[, 1:50],type='class')
reg_log.confusion_matrix <- table(Predicted = reg_log.pred, Actual = data.test$y)
reg_log.confusion_matrix
reg_log.accuracy.test <- sum(reg_log.pred == data.test$y) / length(data.test$y) 
print(reg_log.accuracy.test) # Renvoie 0.62

# QDA 

qda <- qda(y ~ ., data = data.train)
qda.pred <- predict(qda, newdata = data.test)$class
qda.confusion_matrix <- table(Predicted = qda.pred, Actual = data.test$y)
qda.confusion_matrix
qda.accuracy.test <- sum(qda.pred == data.test$y) / length(data.test$y) 
print(qda.accuracy.test) # Renvoie 0.64

# LDA

lda <- lda(y ~ ., data = data.train)
lda.pred <- predict(lda, newdata = data.test)$class
lda.confusion_matrix <- table(Predicted = lda.pred, Actual = data.test$y)
lda.confusion_matrix
lda.accuracy.test <- sum(lda.pred == data.test$y) / length(data.test$y) 
print(lda.accuracy.test) # Renvoie 0.62

# Bayes Naîf

naive_bayes <- naiveBayes(y ~ ., data = data.train)
naive_bayes.pred <- predict(naive_bayes, newdata = data.test)
naive_bayes.confusion_matrix <- table(Predicted = naive_bayes.pred, Actual = data.test$y)
naive_bayes.confusion_matrix
naive_bayes.accuracy.test <- sum(naive_bayes.pred == data.test$y) / length(data.test$y) 
print(naive_bayes.accuracy.test) # Renvoie 0.7

# Test de McNemar

mcnemar.test(lda.pred == data.test$y, qda.pred == data.test$y)

# SVM radial

data.train_scaled <- data.frame(y = data.train$y, data.train.x_scale)
data.test_scaled <- data.frame(y = data.test$y, data.test.x_scale)

svm_model <- svm(y ~ ., data = data.train_scaled, type = "C-classification", kernel = "radial", cross = 10)
svm.pred <- predict(svm_model, newdata = data.test_scaled)
svm.confusion_matrix <- table(Predicted = svm.pred, Actual = data.test_scaled$y)
print(svm.confusion_matrix)
svm.accuracy.test <- sum(svm.pred == data.test_scaled$y) / length(data.test_scaled$y)
print(svm.accuracy.test) #0.63

# SVM linear

data.train_scaled <- data.frame(y = data.train$y, data.train.x_scale)
data.test_scaled <- data.frame(data.test.x_scale, y = data.test$y)

svm_model <- svm(y ~ ., data = data.train_scaled, type = "C-classification", kernel = "linear", cross = 10)
svm.pred <- predict(svm_model, newdata = data.test_scaled)
svm.confusion_matrix <- table(Predicted = svm.pred, Actual = data.test_scaled$y)
print(svm.confusion_matrix)
svm.accuracy.test <- sum(svm.pred == data.test_scaled$y) / length(data.test_scaled$y)
print(svm.accuracy.test) #0.55


# Tree

tree_model <- rpart(y~., data = data.train, method = "class", parms = list(split = 'gini'))
rpart.plot(tree_model, box.palette="RdBu", shadow.col="gray",
           fallen.leaves=FALSE)
plotcp(tree_model)


# Random Forest

data <- read.csv("a24_clas_app.txt", sep = " ")
data$y <- factor(data$y)

rf_model <- randomForest(x = data[, -which(names(data) == "y")], y = as.factor(data$y), 
                         ntree = 500, 
                         mtry = floor(sqrt(ncol(data))),
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
screeplot(acp_result, type = "lines", main = "Scree Plot des 20 premiers axes", npcs = 20)

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

K <- 10  
fold <- sample(1:K, length(train), replace = TRUE)
table(fold)

mean_accuracy <- rep(0, p)

for (nb_axes in 1:p) {
  accuracy <- rep(0, K)
  for (k in 1:K) {
    glm_pca <- multinom(y ~ ., data = train_data_pca[fold != k, c(1:nb_axes, ncol(train_data_pca))])
    pred <- predict(glm_pca, newdata = train_data_pca[fold == k, c(1:nb_axes, ncol(train_data_pca))], type = "class")
    accuracy[k] <- sum(pred == data.test_scaled$y) / length(data.test_scaled$y)
  }
  mean_accuracy[nb_axes] <- mean(accuracy)
}

# Nombre optimal de composantes principales
best_nb_axes <- which.max(mean_accuracy)
cat(sprintf("Le nombre optimal de composantes principales est : %i\n", best_nb_axes))

# GMM
GMM <- MclustDA(x.train, y.train)
pred <- predict(GMM, x.test, type = "class")$classification
print(mean((pred == y.test)))


# Entrainement des modèles

compute_metrics <- function(y_true, y_pred, metric = "accuracy") {
  
  if (metric == "accuracy") {
    # Accuracy calculation
    return(mean(y_pred == y_true))
    
  } else if (metric == "f1_score") {
    return(F1_Score(y_true, y_pred))
  }
}


estimate_metric <- function(model, data.train, metric = "accuracy", rescale = TRUE, k_folds = 10, knn_k = 15, n_components = NULL) {
  set.seed(20241108)
  
  # Rescale the data
  if (rescale == TRUE) {
    data.train <- data.frame(scale(data.train[, 1:50]), y = as.factor(data.train$y))
  }
  
  # Number of folds
  K <- k_folds
  
  # Create fold assignments
  fold <- sample(1:K, nrow(data.train), replace = TRUE)
  
  # Initialize metric values
  metric.val <- rep(0, K)
  
  for (k in 1:K) {
    # Split data into training and validation sets for fold k
    train_data <- data.train[fold != k, ]
    val_data <- data.train[fold == k, ]
    y_train <- as.factor(train_data$y)
    y_val <- as.factor(val_data$y)
    x_train <- train_data[, -which(names(train_data) == "y")]
    x_val <- val_data[, -which(names(val_data) == "y")]
    
    # Apply PCA if n_components is specified
    if (!is.null(n_components) && n_components != 0) {
      # Apply PCA on the training data only
      pca <- prcomp(x_train, center = TRUE, scale. = TRUE)
      
      # Transform both training and validation data using the same PCA object
      x_train <- predict(pca, x_train)[, 1:n_components]
      x_val <- predict(pca, x_val)[, 1:n_components]
      
      # Create updated training and validation data frames
      train_data <- data.frame(y = y_train, x_train)
      val_data <- data.frame(y = y_val, x_val)
    }
    
    # Model fitting and prediction
    if (model == "reg_log") {
      fit <- multinom(y ~ ., data = train_data)
      pred <- predict(fit, newdata = val_data, type = 'class')
      
    } else if (model == "lda") {
      fit <- lda(y ~ ., data = train_data)
      pred <- predict(fit, newdata = val_data)$class
      
    } else if (model == "qda") {
      fit <- qda(y ~ ., data = train_data)
      pred <- predict(fit, newdata = val_data)$class
      
    } else if (model == "naive_bayes") {
      fit <- naiveBayes(y ~ ., data = train_data)
      pred <- predict(fit, newdata = val_data)
      
    } else if (model == "knn") {
      pred <- knn(train = x_train, test = x_val, cl = y_train, k = knn_k)
      
    } else if (model == "gam") {
      fit <- gam(y ~ s(.), data = train_data, family = binomial)
      pred <- as.factor(ifelse(predict(fit, newdata = val_data, type = "response") > 0.5, levels(y_train)[2], levels(y_train)[1]))
      
    } else if (model == "svm_linear") {
      fit <- svm(y ~ ., data = train_data, kernel = "linear", type = "C-classification", cross = 10)
      pred <- predict(fit, newdata = val_data)
      
    } else if (model == "svm_radial") {
      fit <- svm(y ~ ., data = train_data, kernel = "radial", type = "C-classification", cross = 10)
      pred <- predict(fit, newdata = val_data)
      
    } else if (model == "tree") {
      fit <- rpart(y ~ ., data = train_data, method = "class")
      optimal_cp <- fit$cptable[which.min(fit$cptable[,"xerror"]), "CP"]
      fit <- prune(fit, cp = optimal_cp)
      pred <- predict(fit, newdata = val_data, type = "class")
      
    } else if (model == "random_forest") {
      fit <- randomForest(y ~ ., data = train_data)
      pred <- predict(fit, newdata = val_data)
      
    } else if (model == "GMM") {
      fit <- MclustDA(x_train, y_train)
      pred <- predict(fit, x_val)$classification
      
    } else {
      stop("Modèle non reconnu.")
    }
    
    # Calculate the desired metric
    metric.val[k] <- compute_metrics(y_val, pred, metric)
  }
  
  # Return the mean and standard deviation of the metric across all folds
  c(mean(metric.val), sd(metric.val))
}


# Comparaison des modèles de classification avec toutes les composantes

summary_metric_models <- function(data, metric = "accuracy", components_list_models = rep(NULL, 10)) {
  print("COMPARAISON DES MODELES DE CLASSIFICATION :")
  print(paste(metric, " :"))
  
  metric.reg_log <- estimate_metric("reg_log", data, metric = metric, n_components = components_list_models[1])
  print(paste("Reg Log : Mean =", metric.reg_log[1], ", Std Dev =", metric.reg_log[2]))
  
  metric.lda <- estimate_metric("lda", data, metric = metric, n_components = components_list_models[2])
  print(paste("LDA : Mean =", metric.lda[1], ", Std Dev =", metric.lda[2]))
  
  metric.qda <- estimate_metric("qda", data, metric = metric, n_components = components_list_models[3])
  print(paste("QDA : Mean =", metric.qda[1], ", Std Dev =", metric.qda[2]))
  
  metric.naive_bayes <- estimate_metric("naive_bayes", data, metric = metric, n_components = components_list_models[4])
  print(paste("Naive Bayes : Mean =", metric.naive_bayes[1], ", Std Dev =", metric.naive_bayes[2]))
  
  metric.knn <- estimate_metric("knn", data, metric = metric, knn_k = 5, n_components = components_list_models[5])
  print(paste("KNN : Mean =", metric.knn[1], ", Std Dev =", metric.knn[2]))
  
  metric.svm_linear <- estimate_metric("svm_linear", data, metric = metric, n_components = components_list_models[6])
  print(paste("SVM Linear : Mean =", metric.svm_linear[1], ", Std Dev =", metric.svm_linear[2]))
  
  metric.svm_radial <- estimate_metric("svm_radial", data, metric = metric, n_components = components_list_models[7])
  print(paste("SVM Radial : Mean =", metric.svm_radial[1], ", Std Dev =", metric.svm_radial[2]))
  
  metric.tree <- estimate_metric("tree", data, metric = metric, n_components = components_list_models[8])
  print(paste("Arbre de décision : Mean =", metric.tree[1], ", Std Dev =", metric.tree[2]))
  
  metric.random_forest <- estimate_metric("random_forest", data, metric = metric, n_components = components_list_models[9])
  print(paste("Random Forest : Mean =", metric.random_forest[1], ", Std Dev =", metric.random_forest[2]))
  
  metric.gmm <- estimate_metric("GMM", data, metric = metric,, n_components = components_list_models[10])
  print(paste("GMM : Mean =", metric.gmm[1], ", Std Dev =", metric.gmm[2]))
}

summary_metric_models(data, "accuracy")
summary_metric_models(data, "f1_score")


# Choix du nombre de composantes ACP pour entrainer le modèle
plot_pca_components_model <- function(model, data, metric = "accuracy"){
  list_components <- 2:50
  list_avg_metric <- rep(0, 49)
  list_std_metric <- rep(0, 49)
  for (n_components in list_components){
    temp <- estimate_metric(model, data, metric, n_components = n_components)
    list_avg_metric[n_components - 1] <- temp[1]
    list_std_metric[n_components - 1] <- temp[2]
  }
  plot(2:50, list_avg_metric, type = "l", main = paste("PCA", metric, "results with", model), xlab = "Nombre de composantes", ylab = metric)
  max_accuracy_val <- max(list_avg_metric)
  max_index <- which.max(list_avg_metric) + 1
  abline(v = max_index, col = "red", lty = 2)
  print(sprintf("Le nombre de composantes optimal est : %d", max_index))
  return(max_index)
}

components.reg_log <- plot_pca_components_model("reg_log", data, metric = "accuracy") #nb_components = 8
components.lda <-plot_pca_components_model("lda", data, metric = "accuracy") #nb_components = 15
components.qda <-plot_pca_components_model("qda", data, metric = "accuracy") #nb_components = 25
components.naive_bayes <-plot_pca_components_model("naive_bayes", data, metric = "accuracy") #nb_components = 35
components.knn <-plot_pca_components_model("knn", data, metric = "accuracy") #nb_components = 14
components.svm_linear <-plot_pca_components_model("svm_linear", data, metric = "accuracy") #nb_components = 18
components.svm_radial <-plot_pca_components_model("svm_radial", data, metric = "accuracy") #nb_components = 22
components.tree <- plot_pca_components_model("tree", data, metric = "accuracy") #nb_components = 10
componenents.random_forest <- 0 
components.gmm <- 0

# Apprentissage de chaque modèle avec le nombre optimal de composantes ACP
best_components_list <- c(8, 15, 25, 35, 14, 18, 22, 10, 0, 0)
summary_metric_models(data, "accuracy", best_components_list)
























library(farff)
library(dplyr)
library(corrplot)
library(glmnet)
library(mgcv)
library(caret)
library(leaps)
library(gam)
library(mclust)
library(Matrix)
library(corrplot)
library(e1071) 
library(caTools) 
library(class) 
library(nnet)
library(MASS)
library(stats)
library(rpart)
library(rpart.plot)
library(randomForest)
library(MLmetrics)
library(neuralnet)
library(smotefamily)


setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/Projet")
data <- read.csv("a24_clas_app.txt", sep = " ")
data <- data[data$y != 1, ]
data$y <- factor(data$y)

levels(data$y) <- make.names(levels(data$y))

#onehot_vars <- c("X46", "X47", "X48", "X49", "X50")
#scale_vars <- setdiff(names(data), c("y", onehot_vars))

# Preprocess the data
#data <- preprocess_data(data, onehot_vars, scale_vars)
smote <- SMOTE(X = data[, -which(names(data) == "y")], target = data$y, K = 5, dup_size = 2)
data <- data.frame(smote$data)

clas.formula_significant <- as.formula("class ~ X20 + X21 + X22 +X23 + X24 + X25 + X26 + X27 + X28 + X29 +
        X30 + X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 + X40 + X41 + X42 + X43 + X44 + X45 + X46 + X47 + X48 + X49 + X50")


# Test du meilleur modèle avec caret

train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
)

# Initialiser une liste pour les résultats de chaque modèle
results <- list()

# 1. Régression Logistique
grid <- expand.grid(
  decay = c(0, 0.01, 0.1, 1)  # Regularization parameter
)
model_multinom <- train(
  clas.formula_significant, data = data, method = "multinom",
  trControl = train_control, tuneGrid = grid
)
results[["Logistic Regression Multinomial"]] <- model_multinom

# 2. Analyse Discriminante Linéaire (LDA)
model_lda <- train(
  clas.formula_significant, data = data, method = "lda",
  trControl = train_control
)
results[["LDA"]] <- model_lda

# 3. QDA
model_qda <- train(
  clas.formula_significant, data = data, method = "qda",
  trControl = train_control
)
results[["QDA"]] <- model_qda

# 4. Random Forest
grid <- expand.grid(
  mtry = seq(1, 20, by = 1)  # Adjust based on the number of predictors in your data
)
model_rf <- train(
  clas.formula_significant, data = data, method = "rf",
  trControl = train_control, ntree = 100, tuneGrid = grid
)
results[["Random Forest"]] <- model_rf

# 5. Arbre de Décision avec élagage (CART)
grid_rpart <- expand.grid(cp = seq(0, 0.1, 0.01))
model_rpart <- train(
  clas.formula_significant, data = data, method = "rpart",
  trControl = train_control, tuneGrid = grid_rpart
)
results[["Decision Tree (Pruned)"]] <- model_rpart

# 6. Modèles de Mélange Gaussien (GMM)

folds <- createFolds(data$y, k = 10, list = TRUE)
accuracies <- c()

for (i in seq_along(folds)) {
  # Split the data
  train_indices <- setdiff(seq_len(nrow(data)), folds[[i]])
  train_data <- data[train_indices, ]
  test_data <- data[folds[[i]], ]

  # Train the model
  model <- MclustDA(train_data[, -which(names(data) == "y")], class = train_data$y)

  # Make predictions
  predictions <- predict(model, test_data[, -which(names(test_data) == "y")])$classification

  # Calculate accuracy
  accuracy <- mean(predictions == test_data$y)
  accuracies <- c(accuracies, accuracy)
}

# Report cross-validation results
cat("Mean accuracy:", mean(accuracies), "\n")
cat("Accuracy for each fold:", accuracies, "\n")

# 7. Naive Bayes
model_naive_bayes <- train(
  clas.formula_significant, data = data, method = "naive_bayes",
  trControl = train_control
)
results[["Naive Bayes"]] <- model_naive_bayes


# 8. SVM Radial
grid <- expand.grid(
  C = c(0.01, 0.1, 1, 10, 100),
  sigma = c(0.01, 0.01, 0.1, 1)
)

model_svm_radial <- train(
  clas.formula_significant,
  data = data,
  method = "svmRadial",
  tuneGrid = grid
)

results[["SVM radial"]] <- model_svm_radial


# 9. K-Nearest Neighbors (KNN)
grid <- expand.grid(
  k = seq(1, 50, by = 2)
)

model_knn <- train(
  clas.formula_significant, data = data, method = "knn", tuneGrid = grid
)
results[["KNN"]] <- model_knn

# 10. SVM Linear
grid <- expand.grid(
  C = c(0.001, 0.01, 0.1, 1, 10, 100)
)
model_svm_linear <- train(
  clas.formula_significant,
  data = data,
  method = "svmLinear"
)

results[["SVM linear"]] <- model_svm_linear


for (model_name in names(results)) {
  cat("Meilleure précision pour", model_name, ":\n")

  # Get the best accuracy metric
  best_metric <- max(results[[model_name]]$results$Accuracy, na.rm = TRUE)

  # Print the best accuracy
  cat("Précision :", round(best_metric, 4), "\n")
  cat("-------------------\n")
}

library(kernlab)
library(nnet)         # Pour `multinom`
library(MASS)         # Pour `qda`
library(randomForest) # Pour `randomForest`

set.seed(123)

clas.formula_significant_multinom <- as.formula("y ~ X1 + X4 + X6 + X13 + X15 + X17 + X19 + X22 + X23 + X25 + X29 +
                                            X31 + X33 + X35 + X37 + X39 + X41 + X43 + X45 + X46 + 
                                            X47 + X48 + X49 + X50")

clas.formula_significant_svm <- as.formula("y ~ X50 + X49 + X47 + X48 + X46 + X41 + X23 + X43 + X35 + X22 + X25 + X26 + X31")

clas.formula_significant_qda <- as.formula("y ~ X20 + X21 + X22 +X23 + X24 + X25 + X26 + X27 + X28 + X29 +
        X30 + X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 + X40 + X41 + X42 + X43 + X44 + X45")

clas.formula_significant_lda <- as.formula("y ~ X20 + X21 + X22 +X23 + X24 + X25 + X26 + X27 + X28 + X29 +
        X30 + X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 + X40 + X41 + X42 + X43 + X44 + X45")

clas.formula_significant_nb <- as.formula("y ~ X20 + X21 + X22 +X23 + X24 + X25 + X26 + X27 + X28 + X29 +
        X30 + X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 + X40 + X41 + X42 + X43 + X44 + X45")

clas.formula_significant_rf <- as.formula("y ~ X6 + X12 + X13 + X15 + X19 + X22 + X24 + X25 + X26 + X27 + X28 + X29 +
                                            X31 + X32 + X33 + X35 + X36 + X37 + X39 + X41 + X43 + X44 + X46 + 
                                            X47 + X48 + X49 + X50")


# Créer de nouvelles partitions pour la validation croisée interne

inner_fold <- sample(1:K, nrow(data), replace = TRUE)  # `n` doit être défini (nombre d'observations)
inner_accuracy <- numeric(K)
pred_cum <- data.frame()

for (j in 1:K) {
  # Diviser les données selon les folds internes
  train <- data[inner_fold != j, ]
  test <- data[inner_fold == j, ]
  
  # Ajuster les modèles
  model_multinom <- glm(clas.formula_significant_multinom, data = train, family = "binomial")
  pred1 <- predict(model_multinom, newdata = test)  # Prédictions pour `multinom`
  
  model_qda <- qda(clas.formula_significant_qda, data = train)
  pred2 <- predict(model_qda, newdata = test)$class  # Prédictions pour `qda`
  
  model_rf <- randomForest(clas.formula_significant_rf, data = train)
  pred3 <- predict(model_rf, newdata = test)  # Prédictions pour `randomForest`

  model_nb <- naiveBayes(clas.formula_significant_nb, data = train)
  pred4 <- predict(model_nb, newdata = test)  # Prédictions pour `randomForest`
  
  model_svm <- svm(clas.formula_significant_svm, data = train, kernel = "radial", type = "C-classification")
  pred5 <- predict(model_svm, newdata = test)
  
  # Appliquer une règle de majorité
  predictions <- data.frame(pred1, pred2, pred3, pred4, pred5)
  majority_vote <- apply(predictions, 1, function(x) {
    names(which.max(table(x)))  # Trouver la classe majoritaire
  })
  predictions$y <- test$y
  pred_cum <- rbind(pred_cum, predictions)
  
  # Calcul de l'accuracy pour le fold interne
  inner_accuracy[j] <- mean(majority_vote == test$y)
}

# Stocker la moyenne des précisions internes
cv_accuracy <- mean(inner_accuracy)

# Afficher les précisions validées par cross-validation
print(cv_accuracy)




# Générer toutes les combinaisons possibles de modèles
model_combinations <- unlist(lapply(1:(ncol(pred_cum) - 1), function(k) {  # Exclure la colonne cible 'y'
  combn(names(pred_cum)[-ncol(pred_cum)], k, simplify = FALSE)
}), recursive = FALSE)

# Évaluer chaque combinaison
combination_results <- sapply(model_combinations, function(models) {
  # Appliquer la majorité sur les modèles sélectionnés
  majority_vote <- apply(pred_cum[models], 1, function(x) {
    names(which.max(table(x)))  # Classe majoritaire
  })
  
  # Calculer l'accuracy globale pour cette combinaison
  mean(majority_vote == pred_cum$y)
})

# Identifier les meilleures combinaisons
max_accuracy <- max(combination_results)
best_combinations <- model_combinations[which(combination_results == max_accuracy)]

# Afficher les meilleures combinaisons et leur précision
list(
  best_combinations = best_combinations,
  max_accuracy = max_accuracy
)