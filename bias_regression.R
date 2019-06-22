library(recommenderlab)
library(dplyr)

data(MovieLense)
ui_mat <- as(MovieLense, "matrix")

### Find split half correlation for users' ratings
row_lengths <- ui_mat %>% apply(2, function(x) sum((!is.na(x)))) %>% unname()
inds <- which(row_lengths>20 & row_lengths<40)
ui_mat_filt <- ui_mat[,inds ]
ui_list_filt <- apply(ui_mat_filt, 2, function(x) list(x))
ui_list_filt <- lapply(ui_list_filt, function(x) unname(unlist(x)))
ui_list_filt <- lapply(ui_list_filt, function(x) x[!is.na(x)])
ui_sh_list <- lapply(ui_list_filt, function(x) {inds <- sample(1:length(x), 15, replace=F); y <- x[inds]; z <- x[-inds]; c(mean(y), mean(z))})
ui_sh_list %>% as.data.frame() %>% t() %>% cor()
ui_sh_list %>% as.data.frame() %>% t() %>% plot()

### Transpose matrix and turn into binary
iu_mat <- t(ui_mat)
iu_mat_bin <- iu_mat
iu_mat_bin[!is.na(iu_mat_bin)] <- 1
iu_mat_bin[is.na(iu_mat_bin)] <- 0
iu_mat_bin <- rbind(iu_mat_bin, matrix(1, nrow=5, ncol=ncol(iu_mat_bin)))   #add K rows for Bayes/regression
mean_i_means <- mean(rowMeans(iu_mat, na.rm=T), na.rm=T)   #use mean of item means instead of global (for regression point)

### Set up Ax=b and solve regression to get user values, then multiply back and subtract empirical to get item vals
i_means <- rowMeans(iu_mat, na.rm=T) %>% unname()
i_means <- c(i_means, rep(mean_i_means, 5))   #add 5 means for Bayes/regression
u_bias_vec <- solve(t(iu_mat_bin) %*% iu_mat_bin) %*% t(iu_mat_bin) %*% i_means
i_exp_means <- iu_mat_bin %*% u_bias_vec
i_bias_vec <- i_means - i_exp_means
i_bias_vec <- i_bias_vec[1:(length(i_bias_vec)-5)]   #remove the added 5 rows
u_bias_mat <- matrix(as.vector(u_bias_vec), nrow=nrow(ui_mat), ncol=ncol(ui_mat))
i_bias_mat <- matrix(i_bias_vec, nrow=nrow(ui_mat), ncol=ncol(ui_mat), byrow=T)

ui_mat_pred <- u_bias_mat + i_bias_mat + mean_i_means
ui_mat_pred[ui_mat_pred<1] <- 1
ui_mat_pred[ui_mat_pred>5] <- 5

### Create test and train indexes
pct_test <- 0.2
n_test <- round( pct_test * sum(!is.na(ui_mat)), 0)
na_ind <- which(is.na(ui_mat))
test_ind <- sample((1:length(ui_mat))[-na_ind], n_test, replace = F)
train_ind <- (1:length(ui_mat))[-c(na_ind, test_ind)]

### Break matrix into test and train
train_mat <- ui_mat
train_mat[test_ind] <- NA
test_mat <- ui_mat
test_mat[train_ind] <- NA

### Transpose matrix and turn into binary
iu_mat <- t(train_mat)
iu_mat_bin <- iu_mat
iu_mat_bin[!is.na(iu_mat_bin)] <- 1
iu_mat_bin[is.na(iu_mat_bin)] <- 0
iu_mat_bin <- rbind(iu_mat_bin, matrix(1, nrow=5, ncol=ncol(iu_mat_bin)))   #add K rows for Bayes/regression
mean_i_means <- mean(rowMeans(iu_mat, na.rm=T), na.rm=T)   #use mean of item means instead of global (for regression point)

### Set up Ax=b and solve regression to get user values, then multiply back and subtract empirical to get item vals
i_means <- rowMeans(iu_mat, na.rm=T) %>% unname()
i_means <- c(i_means, rep(mean_i_means, 5))   #add 5 means for Bayes/regression
i_means[is.na(i_means)] <- mean_i_means   #impute item means where NA due to missing test data
u_bias_vec <- solve(t(iu_mat_bin) %*% iu_mat_bin) %*% t(iu_mat_bin) %*% i_means
i_exp_means <- iu_mat_bin %*% u_bias_vec
i_bias_vec <- i_means - i_exp_means
i_bias_vec <- i_bias_vec[1:(length(i_bias_vec)-5)]   #remove the added 5 rows
u_bias_mat <- matrix(as.vector(u_bias_vec), nrow=nrow(train_mat), ncol=ncol(train_mat))
i_bias_mat <- matrix(i_bias_vec, nrow=nrow(train_mat), ncol=ncol(train_mat), byrow=T)

train_mat_pred <- u_bias_mat + i_bias_mat + mean_i_means
train_mat_pred[train_mat_pred<1] <- 1
train_mat_pred[train_mat_pred>5] <- 5

### Calculate RMSE on Test data
rmse <- function(predicted, observed, rmna=FALSE){
  sqrt(mean((predicted - observed)^2, na.rm=rmna))
}

rmse_test <- rmse(train_mat_pred, test_mat, rmna=T)
rmse_test
