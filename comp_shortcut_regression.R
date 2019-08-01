# Computational shortcut to linear regression
# (helpful when in very high dimensions and (A^T A) and (A^T b) are easy to manually create)

### Random data
n <- 100
p <- 4
A <- matrix(rnorm(n*p, 1, 0.1), nrow = n, ncol = p)
b <- rnorm(n, 0, 4)

### Solve
cov_Ab <- (1/n) * t(A) %*% b - cbind(colMeans(A)) %*% cbind(mean(b))
var_A <- (1/n) * t(A) %*% A - cbind(colMeans(A)) %*% colMeans(A)
coefs <- solve(var_A, cov_Ab)
#coefs <- solve(cov(A), cov(A, b))   #this is identical to the previous step
slope <- mean(b) - colMeans(A) %*% coefs

### Compare to lm
slope
coefs
summary(lm(b ~ A))$coefficients[,1]
