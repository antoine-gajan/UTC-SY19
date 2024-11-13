# TD4 : Sélection de variables

## Données Residential_building
setwd("C:/Users/antoi/Desktop/UTC/GI05/SY19/TD4")
residential_building <- read.csv2(file = "Residential_Building.csv", sep = ";")

p <- ncol(residential_building)
n <- nrow(residential_building)

# Question 1
hist(residential_building$V9)
hist(residential_building$V10)

residential_building$cost <- log(residential_building$V9)
residential_building$price <- log(residential_building$V10)

# Question 2
library(Matrix)
reg_lin <- lm(price ~. - cost, residential_building)
summary(reg_lin)

p <- 107
X <- as.matrix(residential_building[, 1:p])
rankMatrix(X)

# Question 3
set.seed(1)
K <- 10
fold <- sample(1:K, n, replace = TRUE)
table(fold)

rms <- rep(0, K)
for (k in 1:K){
  cat(sprintf("Processing fold %i\n", k))
  fit <- lm(price ~. - cost, data = residential_building, subset = fold != k)
  pred <- predict(fit, newdata = residential_building[fold == k, ])
  rms[k] <- sqrt(mean((residential_building$price - pred)**2))
}

mean(rms)
sd(rms)

fit <- lm(price ~. - cost, data = residential_building)
plot(fit$fitted.values ~ residential_building$price)

# Question 4
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
  fit <- lm(price ~. - cost, data = residential_building, subset = fold != k)
  fit.aic <- stepAIC(fit)
  fit.bic <- stepAIC(fit, k = log(nrow(residential_building)))
  pred.aic <- predict(fit.aic, newdata = residential_building[fold == k, ])
  pred.bic <- predict(fit.bic, newdata = residential_building[fold == k, ])
  rms.aic[k] <- sqrt(mean((residential_building$price - pred.aic)**2))
  rms.bic[k] <- sqrt(mean((residential_building$price - pred.bic)**2))
}

print(c(mean(rms.aic), sd(rms.aic) / sqrt(K)))
print(c(mean(rms.bic), sd(rms.bic) / sqrt(K)))

print(formula(fit.bic))
reg.fit<-regsubsets(price ~. - cost,data=residential_building,method='exhaustive')
plot(reg.fit,scale="adjr2")
