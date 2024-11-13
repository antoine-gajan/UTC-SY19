# TD7 : Tree-based and ensemble methods

# I) Expenditure and Default dataset

setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/TD7")
expenditure_default <- read.csv("data.csv", sep = ",")
expenditure_default <- expenditure_default[, -c(2,12:14)]

n = nrow(expenditure_default)
p = ncol(expenditure_default)

# 1. Split data
train_index <- sample(1:nrow(expenditure_default), 10000)
expenditure_default.train <- expenditure_default[train_index, ]
expenditure_default.test <- expenditure_default[-train_index, ]

# 2. Classification tree

library(rpart)
library(rpart.plot)

tree_model <- rpart(CARDHLDR ~ ., data = expenditure_default.train, method = "class")
rpart.plot(tree_model, box.palette="RdBu", shadow.col="gray",
           fallen.leaves=FALSE)
plotcp(tree_model)

tree_model.pred_class <- predict(tree_model, newdata = expenditure_default.test, type = "class")
tree_model.pred_prob <- predict(tree_model, newdata = expenditure_default.test, type = "prob")[, 2]

tree_model.conf_mat <- table(tree_model.pred_class, expenditure_default.test$CARDHLDR)
print(tree_model.conf_mat)

error_rate <- mean(tree_model.pred_class != expenditure_default.test$CARDHLDR)
print(error_rate)

# 3. Pruning

printcp(tree_model)
optimal_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]),"CP"]
pruned_tree <- prune(tree_model, cp = optimal_cp)
rpart.plot(pruned_tree)

# 4. ROC Curve

library(pROC)
tree_model.roc <- roc(expenditure_default.test$CARDHLDR, tree_model.pred_prob)
print(tree_model.roc)
plot(tree_model.roc)

# 5. Random forest

x.train = expenditure_default.train[, setdiff(names(expenditure_default.train), "CARDHLDR")]
y.train = as.factor(expenditure_default.train$CARDHLDR)

x.test = expenditure_default.test[, setdiff(names(expenditure_default.train), "CARDHLDR")]
y.test = as.factor(expenditure_default.test$CARDHLDR)

library(randomForest)
rf_model <- randomForest(x = x.train, 
                         y = y.train,
                         xtest = x.test, 
                         ytest = y.test,
                         ntree = 50, 
                         nodesize = 1,
                         importance = TRUE, 
                         keep.forest = TRUE)

print(rf_model)
rf.confusion_matrix <- table(Predicted = rf_model$test$predicted, Actual = y.test)
print(rf.confusion_matrix)

rf.error_rate <- mean(rf_model$test$predicted != y.test)
print(rf.error_rate)

varImpPlot(rf_model)

# 6. Comparison with logistic regression

reg_log <- glm(CARDHLDR ~ ., data = expenditure_default.train, family = "binomial")
reg_log.pred_proba <- predict(reg_log, expenditure_default.test, type = "response")
reg_log.pred <- ifelse(reg_log.pred_proba > 0.5, 1, 0)
reg_log.error_rate <- mean(reg_log.pred != expenditure_default.test$CARDHLDR)
print(reg_log.error_rate)


# II) Boston data

library(MASS)
data(Boston)

train_index <- sample(1:nrow(Boston), 4 * nrow(Boston) / 5)
boston.train <- Boston[train_index, ]
boston.test <- Boston[-train_index, ]


tree_model_2 <- rpart(medv ~., data = Boston, subset = train_index)
rpart.plot(tree_model_2, box.palette="RdBu", shadow.col="gray",
           fallen.leaves=FALSE)
plotcp(tree_model_2)

tree_model_2.pred <- predict(tree_model_2, newdata = boston.test)
tree_model_2.rmse <- sqrt(mean((tree_model_2.pred - boston.test$medv)**2))
print(tree_model_2.rmse)

rf_model <- randomForest(x = boston.train[, setdiff(names(boston.train), "medv")], 
                         y = boston.train$medv,
                         xtest = boston.test[, setdiff(names(boston.test), "medv")], 
                         ytest = boston.test$medv,
                         ntree = 50, 
                         nodesize = 1,
                         importance = TRUE, 
                         keep.forest = TRUE)

print(rf_model)
rf.confusion_matrix <- table(Predicted = rf_model$test$predicted, Actual = y.test)
print(rf.confusion_matrix)

rf_model.rmse <- sqrt(mean((rf_model$test$predicted - boston.test$medv)**2))
print(rf_model.rmse)

varImpPlot(rf_model)