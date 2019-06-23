source('C:/elimir/media/driveb/projects/nhl/Vault/R/Sunny/sunny_useful_libraries.R')
library(recommenderlab)

data(MovieLense)
ui_mat <- as(MovieLense, "matrix")

# ### Find split half correlation for users' ratings
# row_lengths <- ui_mat %>% apply(2, function(x) sum((!is.na(x)))) %>% unname()
# inds <- which(row_lengths>20 & row_lengths<40)
# ui_mat_filt <- ui_mat[,inds ]
# ui_list_filt <- apply(ui_mat_filt, 2, function(x) list(x))
# ui_list_filt <- lapply(ui_list_filt, function(x) unname(unlist(x)))
# ui_list_filt <- lapply(ui_list_filt, function(x) x[!is.na(x)])
# ui_sh_list <- lapply(ui_list_filt, function(x) {inds <- sample(1:length(x), 15, replace=F); y <- x[inds]; z <- x[-inds]; c(mean(y), mean(z))})
# ui_sh_list %>% as.data.frame() %>% t() %>% cor()
# ui_sh_list %>% as.data.frame() %>% t() %>% plot()

### Create binary matrix out of appropriate quadrants
ur_mat <- ui_mat
ur_mat[!is.na(ur_mat)] <- 1
ur_mat[is.na(ur_mat)] <- 0
ul_mat <- matrix(0, nrow=nrow(ui_mat), ncol=nrow(ui_mat))
diag(ul_mat) <- rowSums(ur_mat, na.rm=T)
lr_mat <- matrix(0, nrow=ncol(ui_mat), ncol=ncol(ui_mat))
diag(lr_mat) <- colSums(ur_mat, na.rm=T)
ll_mat <- t(ur_mat)
ui_mat_bin <- rbind(cbind(ul_mat, ur_mat), cbind(ll_mat, lr_mat))

### Create means vectors
u_means_vec <- rowMeans(ui_mat, na.rm=T)
i_means_vec <- colMeans(ui_mat, na.rm=T)
ui_means_vec <- c(u_means_vec, i_means_vec)
mean_u_means <- mean(u_means_vec, na.rm=T)
mean_i_means <- mean(i_means_vec, na.rm=T)
u_sums_vec <- rowSums(ui_mat, na.rm=T)
i_sums_vec <- colSums(ui_mat, na.rm=T)
ui_sums_vec <- c(u_sums_vec, i_sums_vec)

### Set up solvable matrix version of Bayes / ridge regression
K <- 1
A <- ui_mat_bin
diag(A) <- diag(A) + K
b <- unname(ui_sums_vec)
b <- b + c(rep(mean_u_means, length(u_means_vec)), rep(mean_i_means, length(i_means_vec))) * K
x <- solve(A, b)

### Use values to predict
ui_mat_pred <- matrix(NA, nrow = nrow(ui_mat), ncol=ncol(ui_mat))
for (i in 1:nrow(ui_mat_pred)){
  ui_mat_pred[i, ] <- unname(unlist(x[i] + x[(length(u_means_vec)+1):length(x)]))
}
ui_mat_pred[ui_mat_pred<1] <- 1
ui_mat_pred[ui_mat_pred>5] <- 5

# ### Transpose matrix and turn into binary
# iu_mat <- t(ui_mat)
# iu_mat_bin <- iu_mat
# iu_mat_bin[!is.na(iu_mat_bin)] <- 1
# iu_mat_bin[is.na(iu_mat_bin)] <- 0
# iu_mat_bin <- rbind(iu_mat_bin, matrix(1, nrow=5, ncol=ncol(iu_mat_bin)))   #add K rows for Bayes/regression
# mean_i_means <- mean(rowMeans(iu_mat, na.rm=T), na.rm=T)   #use mean of item means instead of global (for regression point)
# 
# ### Set up Ax=b and solve regression to get user values, then multiply back and subtract empirical to get item vals
# i_means <- rowMeans(iu_mat, na.rm=T) %>% unname()
# i_means <- c(i_means, rep(mean_i_means, 5))   #add 5 means for Bayes/regression
# u_bias_vec <- solve(t(iu_mat_bin) %*% iu_mat_bin) %*% t(iu_mat_bin) %*% i_means
# i_exp_means <- iu_mat_bin %*% u_bias_vec
# i_bias_vec <- i_means - i_exp_means
# i_bias_vec <- i_bias_vec[1:(length(i_bias_vec)-5)]   #remove the added 5 rows
# u_bias_mat <- matrix(as.vector(u_bias_vec), nrow=nrow(ui_mat), ncol=ncol(ui_mat))
# i_bias_mat <- matrix(i_bias_vec, nrow=nrow(ui_mat), ncol=ncol(ui_mat), byrow=T)
# 
# ui_mat_pred <- u_bias_mat + i_bias_mat + mean_i_means
# ui_mat_pred[ui_mat_pred<1] <- 1
# ui_mat_pred[ui_mat_pred>5] <- 5

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

### Create binary matrix out of appropriate quadrants
ur_mat <- train_mat
ur_mat[!is.na(ur_mat)] <- 1
ur_mat[is.na(ur_mat)] <- 0
ul_mat <- matrix(0, nrow=nrow(train_mat), ncol=nrow(train_mat))
diag(ul_mat) <- rowSums(ur_mat, na.rm=T)
lr_mat <- matrix(0, nrow=ncol(train_mat), ncol=ncol(train_mat))
diag(lr_mat) <- colSums(ur_mat, na.rm=T)
ll_mat <- t(ur_mat)
train_mat_bin <- rbind(cbind(ul_mat, ur_mat), cbind(ll_mat, lr_mat))

### Create means vectors
u_means_vec <- rowMeans(train_mat, na.rm=T)
i_means_vec <- colMeans(train_mat, na.rm=T)
ui_means_vec <- c(u_means_vec, i_means_vec)
mean_u_means <- mean(u_means_vec, na.rm=T)
mean_i_means <- mean(i_means_vec, na.rm=T)
u_sums_vec <- rowSums(train_mat, na.rm=T)
i_sums_vec <- colSums(train_mat, na.rm=T)
ui_sums_vec <- c(u_sums_vec, i_sums_vec)

### Set up solvable matrix version of Bayes / ridge regression
K <- 1
A <- train_mat_bin
diag(A) <- diag(A) + K
b <- unname(ui_sums_vec)
b <- b + c(rep(mean_u_means, length(u_means_vec)), rep(mean_i_means, length(i_means_vec))) * K
x <- solve(A, b)

### Use values to predict
train_mat_pred <- matrix(NA, nrow = nrow(train_mat), ncol=ncol(train_mat))
for (i in 1:nrow(train_mat_pred)){
  train_mat_pred[i, ] <- unname(unlist(x[i] + x[(length(u_means_vec)+1):length(x)]))
}
train_mat_pred[train_mat_pred<1] <- 1
train_mat_pred[train_mat_pred>5] <- 5

# ### Transpose matrix and turn into binary
# iu_mat <- t(train_mat)
# iu_mat_bin <- iu_mat
# iu_mat_bin[!is.na(iu_mat_bin)] <- 1
# iu_mat_bin[is.na(iu_mat_bin)] <- 0
# iu_mat_bin <- rbind(iu_mat_bin, matrix(1, nrow=5, ncol=ncol(iu_mat_bin)))   #add K rows for Bayes/regression
# mean_i_means <- mean(rowMeans(iu_mat, na.rm=T), na.rm=T)   #use mean of item means instead of global (for regression point)
# 
# ### Set up Ax=b and solve regression to get user values, then multiply back and subtract empirical to get item vals
# i_means <- rowMeans(iu_mat, na.rm=T) %>% unname()
# i_means <- c(i_means, rep(mean_i_means, 5))   #add 5 means for Bayes/regression
# i_means[is.na(i_means)] <- mean_i_means   #impute item means where NA due to missing test data
# u_bias_vec <- solve(t(iu_mat_bin) %*% iu_mat_bin) %*% t(iu_mat_bin) %*% i_means
# i_exp_means <- iu_mat_bin %*% u_bias_vec
# i_bias_vec <- i_means - i_exp_means
# i_bias_vec <- i_bias_vec[1:(length(i_bias_vec)-5)]   #remove the added 5 rows
# u_bias_mat <- matrix(as.vector(u_bias_vec), nrow=nrow(train_mat), ncol=ncol(train_mat))
# i_bias_mat <- matrix(i_bias_vec, nrow=nrow(train_mat), ncol=ncol(train_mat), byrow=T)
# 
# train_mat_pred <- u_bias_mat + i_bias_mat + mean_i_means
# train_mat_pred[train_mat_pred<1] <- 1
# train_mat_pred[train_mat_pred>5] <- 5

### Calculate RMSE on Test data
rmse <- function(predicted, observed, rmna=FALSE){
  sqrt(mean((predicted - observed)^2, na.rm=rmna))
}

rmse_test <- rmse(train_mat_pred, test_mat, rmna=T)
print(rmse_test)

# ### One hot encode for regression
# train_df_tidy <- train_mat %>% as_tibble() %>% 
#   mutate(user=row_number()) %>% 
#   gather(key=item, value=rating, -user) %>% 
#   filter(!is.na(rating))
# train_df_onehot <- train_df_tidy %>% 
#   mutate(id=row_number(), user_bin=1, item_bin=1) %>% 
#   spread(value=user_bin, key=user) %>% 
#   spread(value=item_bin, key=item) %>% 
#   select(-id)
# train_df_onehot[is.na(train_df_onehot)] <- 0
# 
# A <- as.matrix(train_df_onehot %>% rename(intercept = rating) %>% mutate(intercept = 1))
# b <- train_df_onehot$rating
# lm_fit <- .lm.fit(A, b)
