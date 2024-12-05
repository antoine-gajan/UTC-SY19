# TD8


# Partie 1
## Question 1
sample.donut <- function(n1, r1, n2, r2) {
    R1 <- rnorm(n1, mean = r1)
    angle1 <- runif(n1, 0, 2 * pi)
    class1 <- data.frame(X1 = R1 * cos(angle1), X2 = R1 * sin(angle1))

    R2 <- rnorm(n2, mean = r2)
    angle2 <- runif(n2, 0, 2 * pi)
    class2 <- data.frame(X1 = R2 * cos(angle2), X2 = R2* sin(angle2))

    cbind(rbind(class1, class2), y = factor(c(rep(0, n1), rep(1, n2))))
}

n1 <- 500
n2 <- 500
r1 <- 3
r2 <- 5

X <- sample.donut(n1, r1, n2, r2)
plot(X[, 1:2], col = X$y)
plot(X[, 1], (X[, 1])**2 + (X[, 2])**2, col = X$y)

## Question 2

X.train <- sample.donut(n1, r1, n2, r2)
X.test <- sample.donut(10 * n1, r1, 10 * n2, r2)

library(MASS)
lda <- lda(y ~ ., data = X.train)
pred <- predict(lda, newdata = X.test)$class
mean(pred == X.test$y)


## Question 3

lda <- lda(y ~ poly(X1, degree = 2) + poly(X2, degree = 2), data = X.train)
pred <- predict(lda, newdata = X.test)$class
mean(pred == X.test$y)

# Partie 2

## Question 1

library(mclust)
fit.MLDA <- MclustDA(X.train[, -3], X.train$y)
plot(fit.MLDA, what = "scatterplot")

summary(fit.MLDA, newdata = X.test[, -3], newclass = X.test$y)

# Partie 3

## Question 1

library(mgcv)

fit <- gam(y ~ s(X1) + s(X2), family = "binomial", data = X.train)
pred <- ifelse(predict(fit, newdata = X.test, type = "link") > 0, 1, 0)
mean(pred == X.test$y)
table(Predicted = pred, Actual = X.test$y)
summary(fit)

## Question 2

vis.gam(fit, type = "response", plot.type = "contour")
points(X.train)

## Question 3

plot(fit, select = 1)
plot(fit, select = 2)