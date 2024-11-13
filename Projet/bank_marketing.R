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


# Entrainement des modèles


## Train / test dataset

train_index <- sample(1:nrow(bank_marketing), 4 * nrow(bank_marketing) / 5)
bank_marketing.train <- bank_marketing[train_index, ]
bank_marketing.test <- bank_marketing[-train_index, ]

