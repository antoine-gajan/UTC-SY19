library(MASS)
library(e1071) 
library(caTools) 
library(class)

X.reg = read.table("a24_reg_app.txt")
reg <- lm(y ~ ., data = X.reg)
summary_fit <- summary(reg)
p_values <- summary_fit$coefficients[, 4] 
significant_vars <- names(p_values[p_values < 0.05 & names(p_values) != "(Intercept)"])
formula_significant <- as.formula(paste("y ~", paste(significant_vars, collapse = " + ")))
reg <- lm(formula_significant, data = X.reg)


X.clas = read.table("a24_clas_app.txt")
clas <- naiveBayes(y ~ ., data = X.clas)

classifieur <- function(test_set) {
  library(MASS)
  predict(clas, test_set)$class
}

regresseur <- function(test_set) {
  library(MASS)
  predict(reg, test_set)
}

save("clas", "reg", "classifieur", "regresseur", file = "env.Rdata")