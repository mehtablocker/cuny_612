### Stochastic Gradient Descent function (Funk SVD)
library(recommenderlab)
data(MovieLense)
ui_mat <- as(MovieLense, "matrix")

rmse <- function(predicted, observed, rmna=FALSE){
  sqrt(mean((predicted - observed)^2, na.rm=rmna))
}

### Normalize the User-Item matrix
global_mean <- mean(ui_mat, na.rm = T)
global_mean
u_bias_vec <- unname(rowMeans(ui_mat, na.rm=T)) - global_mean
u_bias_mat <- matrix(u_bias_vec, nrow=nrow(ui_mat), ncol=ncol(ui_mat))
i_bias_vec <- unname(colMeans(ui_mat, na.rm=T)) - global_mean
i_bias_mat <- matrix(i_bias_vec, nrow=nrow(ui_mat), ncol=ncol(ui_mat), byrow=T)
baseline <- global_mean + u_bias_mat + i_bias_mat
ui_mat_norm <- ui_mat - baseline

### Set parameters
k <- 2   #number of dimensions to compress to
init <- 0.1   #initial guess for values
lrn <- 0.001   #learning rate, ie., lambda, or step size in descent
reg_gamma <- 0.015   #regularization coefficient
rmse_change <- 0.000001   #stopping point
max_epochs <- 200   #stopping point

U <- matrix(init, nrow=nrow(ui_mat_norm), ncol=k)
V <- matrix(init, nrow=ncol(ui_mat_norm), ncol=k)
old_comp_rmse <- 100
new_comp_rmse <- 99
epoch <- 1

### Loop through matrix to update U and V
while (abs(old_comp_rmse - new_comp_rmse)>rmse_change & epoch<=max_epochs){
  for (f in 1:k){
    ui_mat_norm_comp_sgd <- U %*% t(V)
    errs <- ui_mat_norm - ui_mat_norm_comp_sgd
    for (j in 1:ncol(errs)){
      adj <- lrn * (errs[, j] * t(V)[f, j] - reg_gamma * U[, f])
      adj[is.na(adj)] <- 0
      U[, f] <- U[, f] + adj
    }
    for (i in 1:nrow(errs)){
      adj <- lrn * (errs[i, ] * U[i, f] - reg_gamma * V[, f])
      adj[is.na(adj)] <- 0
      V[, f] <- V[, f] + adj
    }
  }
  ui_mat_norm_comp_sgd <- U %*% t(V)
  old_comp_rmse <- new_comp_rmse
  new_comp_rmse <- rmse(ui_mat_norm_comp_sgd, ui_mat_norm, rmna=T)
  epoch <- epoch + 1
  cat(":")
}
cat("RMSE = ", new_comp_rmse, "; Last RMSE change = ", abs(new_comp_rmse-old_comp_rmse))

### Compare to funkSVD function
fSVD_list <- funkSVD(ui_mat_norm, k = 900, verbose=T)
ui_mat_norm_comp_fsvd <- fSVD_list$U %*% t(fSVD_list$V)
rmse(ui_mat_norm_comp_fsvd, ui_mat_norm, rmna=T)
