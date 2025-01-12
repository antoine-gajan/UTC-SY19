# Installation des packages

install.packages("farff")
install.packages("dplyr")
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

# Lecture du dataset
setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/Projet")
bank_marketing <- readARFF("bank_marketing.arff")


# Preprocessing du dataset

## Renommage des variables
colnames(bank_marketing) <- c(
  "age",        # V1
  "job",        # V2
  "marital",    # V3
  "education",  # V4
  "default",    # V5
  "balance",    # V6
  "housing",    # V7
  "loan",       # V8
  "contact",    # V9
  "day",        # V10
  "month",      # V11
  "duration",   # V12
  "campaign",   # V13
  "pdays",      # V14
  "previous",   # V15
  "poutcome",   # V16
  "y"           # Class
)
bank_marketing$y <- factor(bank_marketing$y)
levels(bank_marketing$y) <- make.names(levels(bank_marketing$y))

preprocess_data_frame <- function(df, rescale = TRUE) {
  target_var <- "y"
  
  # Separate predictors and target variable
  predictors <- df[, !(names(df) %in% target_var)]  # Exclude 'y'
  target <- df[[target_var]]
  
  # Rescale the data
  if (rescale == TRUE) {
    numeric_predictors_columns <- sapply(df, is.numeric)
    df[, numeric_predictors_columns] <- scale(df[, numeric_predictors_columns])
  }
  
  # One-hot encode all factor variables in predictors
  encoded_predictors <- model.matrix(~ . + 0, data = predictors)
  
  # Combine the encoded predictors with the target variable
  final_data <- data.frame(encoded_predictors, y = target)
  return(final_data)
}



rf_model <- randomForest(x = bank_marketing[, -which(names(bank_marketing) == "y")], y = as.factor(bank_marketing$y), 
                         ntree = 500, 
                         mtry = floor(sqrt(ncol(bank_marketing))),
                         nodesize = 1,
                         importance = TRUE, 
                         keep.forest = TRUE)

importance_scores <- importance(rf_model, type = 1)  # type = 1 gives Mean Decrease in Accuracy



varImpPlot(rf_model)



set.seed(20241108)

# Data analysis

## Corrplot
numeric_columns <- sapply(bank_marketing, is.numeric)
numeric_data <- bank_marketing[, numeric_columns]
cor_matrix <- cor(numeric_data, use = "complete.obs")

corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black",
         number.cex = 0.7)

par(mfrow = c(2, 2))
for (numeric_var in colnames(numeric_data)[1:4]) {
  hist(numeric_data[[numeric_var]], xlab = numeric_var, ylab = "Count", col = "lightblue", main = paste("Histogram of", numeric_var))
}

for (numeric_var in colnames(numeric_data)[5:7]) {
  hist(numeric_data[[numeric_var]], xlab = numeric_var, ylab = "Count", col = "lightblue", main = paste("Histogram of", numeric_var))
}


## Factor data
factor_columns <- sapply(bank_marketing, is.factor)  # Use is.factor to identify factor columns
factor_data <- bank_marketing[, factor_columns]


par(mfrow = c(3, 2))
# Generate bar plots for each factor variable
for (factor_var in colnames(factor_data)[1:6]) {
  counts <- table(factor_data[[factor_var]])
  barplot(counts, main = paste("Distribution of", factor_var), 
          xlab = factor_var, ylab = "Count", col = "lightblue")
}

par(mfrow = c(2, 2))
# Generate bar plots for each factor variable
for (factor_var in colnames(factor_data)[7:10]) {
  counts <- table(factor_data[[factor_var]])
  barplot(counts, main = paste("Distribution of", factor_var), 
          xlab = factor_var, ylab = "Count", col = "lightblue")
}

# Distribution of y
par(mfrow = c(1, 1))
counts <- table(factor_data[["y"]])
barplot(counts, main = paste("Distribution of y"), 
        xlab = "y", ylab = "Count", col = "lightblue")

par(mfrow = c(1, 1))

# ACP 
data_standardized <- scale(numeric_data[, !names(numeric_data) %in% 'y'])
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

# Représentation en 2 dimensions
pc_data <- as.data.frame(acp_result$x[, 1:2])  # Les deux premières composantes principales
y <- bank_marketing$y  # Variable cible

library(ggplot2)

ggplot(pc_data, aes(x = PC1, y = PC2, color = y)) +
  geom_point(size = 1) +
  labs(title = "Représentation des données en 2 dimensions",
       x = "Première composante principale (PC1)",
       y = "Deuxième composante principale (PC2)",
       color = "Valeur de y") + xlim(-5, 20) + ylim(-10, 15)
  theme_minimal()


# Split train test
train_indices <- sample(1:nrow(bank_marketing), size = round(0.8 * nrow(bank_marketing)))
bank_marketing.train <- bank_marketing[train_indices, ]
bank_marketing.train <- downSample(x = bank_marketing.train[, -which(names(bank_marketing.train) == "y")], 
                            y = bank_marketing.train$y)
bank_marketing.train$y <- bank_marketing.train$Class
bank_marketing.train$Class <- NULL

bank_marketing.test <- bank_marketing[-train_indices, ]

library(dplyr)

f1_score_summary <- function(data, lev = NULL, model = NULL) {
  cm <- confusionMatrix(data$pred, data$obs)
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  c(F1 = f1)
}

# Create trainControl with custom F1 score metric
train_control <- trainControl(
  method = "cv", 
  number = 10, 
  summaryFunction = f1_score_summary,  # Use twoClassSummary for binary classification
  classProbs = TRUE, 
  savePredictions = "final"
)

# Initialiser une liste pour les résultats de chaque modèle
results <- list()

# 1. Régression Logistique
model_multinom <- train(
  y ~ ., data = bank_marketing.train, method = "multinom", 
  trControl = train_control
)
results[["Logistic Regression Multinomial"]] <- model_multinom

# 2. Analyse Discriminante Linéaire (LDA)
model_lda <- train(
  y ~ ., data = bank_marketing.train, method = "lda",
  trControl = train_control
)
results[["LDA"]] <- model_lda

# 3. QDA
model_qda <- train(
  y ~ ., data = bank_marketing.train, method = "qda",
  trControl = train_control
)
results[["QDA"]] <- model_qda

# 4. Random Forest
model_rf <- train(
  y ~ ., data = bank_marketing.train, method = "rf",
  trControl = train_control, ntree = 50
)
results[["Random Forest"]] <- model_rf

# 5. Arbre de Décision avec élagage (CART)
grid_rpart <- expand.grid(cp = seq(0, 0.1, 0.01))
model_rpart <- train(
  y ~ ., data = bank_marketing.train, method = "rpart",
  trControl = train_control, tuneGrid = grid_rpart
)
results[["Decision Tree (Pruned)"]] <- model_rpart

# 6. Modèles de Mélange Gaussien (GMM)

model_gmm <- MclustDA(bank_marketing.train[, -which(names(bank_marketing.train) == "y")], class = bank_marketing.train$y)

# 7. Naive Bayes
model_naive_bayes <- train(
  y ~ ., data = bank_marketing.train, method = "naive_bayes",
  trControl = train_control
)
results[["Naive Bayes"]] <- model_naive_bayes


# 8. SVM Radial
model_svm_radial <- train(
  y ~ ., 
  data = bank_marketing.train, 
  method = "svmRadial"
)

results[["SVM radial"]] <- model_svm_radial


# 9. K-Nearest Neighbors (KNN)
model_knn <- train(
  y ~ ., data = bank_marketing.train, method = "knn"
)
results[["KNN"]] <- model_knn

# 10. SVM Linear avec 10% des données
model_svm_linear <- train(
  y ~ ., 
  data = bank_marketing.train, 
  method = "svmLinear"
)

results[["SVM linear"]] <- model_svm_linear

# Afficher les résultats de chaque modèle
for (model_name in names(results)) {
  cat("Résultats pour", model_name, ":\n")
  print(results[[model_name]])
  cat("\n-------------------\n")
}

# Comparer les performances des modèles
resamps <- resamples(results)
summary(resamps)
bwplot(resamps)


# Enregistrer les modèles

# Dossier pour enregistrer les modèles
output_dir <- "saved_models_one_hot"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Créer une fonction pour enregistrer chaque modèle
save_best_model <- function(model, model_name) {
  file_path <- file.path(output_dir, paste0(model_name, ".rds"))
  saveRDS(model, file_path)
  cat("Modèle", model_name, "enregistré dans", file_path, "\n")
}

save_best_model(model_multinom, "Logistic_Regression_Multinomial")
save_best_model(model_lda, "LDA")
save_best_model(model_qda, "QDA")
save_best_model(model_rf, "Random_Forest")
save_best_model(model_rpart, "Decision_Tree_Pruned")
save_best_model(model_knn, "KNN")
save_best_model(model_naive_bayes, "Naive_Bayes")
save_best_model(model_svm_radial, "SVM_Radial")
save_best_model(model_svm_linear, "SVM_Linear")
save_best_model(model_gmm, "GMM")


# Chargement des modèles enregistrés

# Charger tous les modèles sauvegardés une seule fois
load_saved_models <- function(saved_models_dir) {
  model_files <- list.files(saved_models_dir, pattern = "\\.rds$", full.names = TRUE)
  loaded_models <- list()
  for (file_path in model_files) {
    model_name <- gsub("\\.rds$", "", basename(file_path))
    loaded_models[[model_name]] <- readRDS(file_path)
    cat("Modèle", model_name, "chargé avec succès.\n")
  }
  return(loaded_models)
}


calculate_classification_score <- function(y_true, y_pred, metric = "accuracy") {
  tryCatch({
    if (metric == 'accuracy') {
      return(mean(y_true == y_pred))
    }
    else if (metric == "f1_score") {
      # Create confusion matrix
      confusion_matrix <- table(Actual = y_true, Predicted = y_pred)
      TP <- confusion_matrix[2, 2]  
      FP <- confusion_matrix[1, 2]  
      FN <- confusion_matrix[2, 1]
      
      Precision <- TP / (TP + FP)
      Recall <- TP / (TP + FN)
      
      F1 <- 2 * (Precision * Recall) / (Precision + Recall)
      return(F1)
    }
  }, error = function(e) {
    # If there's an error (e.g., divide by zero), return 0
    return(0)
  })
}


# Calculer les métriques de classification avec les modèles déjà chargés
estimate_classification_metric <- function(model, test_data, loaded_models = list()) {
  K <- 10  # Nombre de folds
  set.seed(20240811)
  
  # Vérifier si le modèle est déjà chargé
  if (!(model %in% names(loaded_models))) {
    stop(paste("Le modèle", model, "n'est pas chargé."))
  }
  
  loaded_model <- loaded_models[[model]]
    
  pred <- predict(loaded_model, newdata = test_data)
    
  acc <- calculate_classification_score(test_data$y, pred, "accuracy")
  f1 <- calculate_classification_score(test_data$y, pred, "f1_score")
  
  # Retourner un vecteur nommé
  return(c(
    accuracy = acc, 
    f1 = f1
  ))
}

library(tibble) # Pour construire des tableaux structurés
library(dplyr)  # Pour trier les données

summary_metric_models <- function(data, saved_models_dir = NULL) {

  # Charger tous les modèles sauvegardés
  loaded_models <- load_saved_models(saved_models_dir)
  
  # Initialiser un tableau pour stocker les résultats
  results <- data.frame(
    Model = character(),
    F1 = numeric(),
    Accuracy = numeric()
  )

  # Boucle sur les modèles pour calculer les métriques
  for (model in names(loaded_models)) {
    if (model != "GMM") {
      # Évaluer le modèle
      result <- estimate_classification_metric(
        model = model,
        test_data = data,
        loaded_models = loaded_models
      )
      # Ajouter les résultats au tableau
      results <- rbind(results, data.frame(
        Model = model,
        F1 = as.numeric(result["f1"]),
        Accuracy = as.numeric(result["accuracy"])
      ))
    }
  }
  
  # Trier les résultats par F1_mean décroissant
  results <- results %>%
    arrange(desc(F1))
  
  return(as_tibble(results))
}


# Evaluation des modèles
saved_models_dir <- "saved_models"
loaded_models <- load_saved_models(saved_models_dir)
summary_metric_models(data = bank_marketing.test, saved_models_dir = saved_models_dir)

saved_models_dir <- "saved_models_undersampling"
loaded_models <- load_saved_models(saved_models_dir)
summary_metric_models(data = bank_marketing.test, saved_models_dir = saved_models_dir)

saved_models_dir <- "saved_models_one_hot"
loaded_models <- load_saved_models(saved_models_dir)
summary_metric_models(data = bank_marketing.test, saved_models_dir = saved_models_dir)



# Mise en place d'un réseau de neurones

library(keras)
library(caret)




bank_marketing <- readARFF("bank_marketing.arff")
colnames(bank_marketing) <- c(
  "age",        # V1
  "job",        # V2
  "marital",    # V3
  "education",  # V4
  "default",    # V5
  "balance",    # V6
  "housing",    # V7
  "loan",       # V8
  "contact",    # V9
  "day",        # V10
  "month",      # V11
  "duration",   # V12
  "campaign",   # V13
  "pdays",      # V14
  "previous",   # V15
  "poutcome",   # V16
  "y"           # Class
)

bank_marketing <- preprocess_data_frame(bank_marketing)
bank_marketing.train <- preprocess_data_frame(bank_marketing.train)
bank_marketing.test <- preprocess_data_frame(bank_marketing.test)

# Split train test
train_indices <- sample(1:nrow(bank_marketing), size = round(0.8 * nrow(bank_marketing)))
bank_marketing.train <- preprocess_data_frame(bank_marketing.train)
bank_marketing.test <- preprocess_data_frame(bank_marketing.test)

X_train <- as.matrix(bank_marketing.train[, -ncol(bank_marketing.train)])
y_train <- as.numeric(as.factor(bank_marketing.train$y)) - 1
X_test <- as.matrix(bank_marketing.test[, -ncol(bank_marketing.test)])
y_test <- as.numeric(as.factor(bank_marketing.test$y)) - 1

# Define the Keras model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")  # Output layer for binary classification

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy"))

# Train the model
model %>% fit(
  X_train, y_train,
  epochs = 20,
  batch_size = 16,
  validation_split = 0.2,
  verbose = 1
)

# Make predictions
pred_prob <- model %>% predict(X_test)
pred_prob
pred <- ifelse(pred_prob > 0.5, 1, 0)

# Create a confusion matrix
conf_matrix <- table(Actual = y_test, Predicted = pred)
print(conf_matrix)

calculate_classification_score(y_test, pred, "f1_score")
calculate_classification_score(y_test, pred, "accuracy")