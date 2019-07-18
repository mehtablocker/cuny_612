### Load libraries
import pandas as pd
import numpy as np
import funk_svd as fsvd
np.set_printoptions(suppress=True)

### Get data
col_names = ["user_id", "item_id", "rating", "timestamp"]
ratings_df = pd.read_csv('https://raw.githubusercontent.com/mehtablocker/cuny_612/master/data_files/MovieLens/u.data', sep='\t', names=col_names, encoding='latin-1')
ui_mat = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating").to_numpy()

### Create test and train indexes
pct_test = 0.2
n_test = int(round(pct_test * np.count_nonzero(~np.isnan(ui_mat)), 0))
non_na_ind = np.argwhere(~np.isnan(ui_mat))
seq = np.arange(len(non_na_ind[:, 0]))
test_ind = np.random.choice(seq, n_test, replace = False)
train_ind = np.delete(seq, test_ind)

### Break matrix into test and train
train_mat = np.copy(ui_mat)
train_mat[non_na_ind[test_ind, 0], non_na_ind[test_ind, 1]] = np.nan
test_mat = np.copy(ui_mat)
test_mat[non_na_ind[train_ind, 0], non_na_ind[train_ind, 1]] = np.nan
train_mat[0:4, 0:4]
test_mat[0:4, 0:4]

### Calculate baseline and normalize matrix
global_mean = np.nanmean(train_mat)
u_bias_vec = np.nanmean(train_mat, axis = 1) - global_mean
u_bias_mat = np.tile(u_bias_vec, (len(train_mat[0,:]), 1)).transpose()
i_bias_vec = np.nanmean(train_mat, axis = 0) - global_mean
i_bias_mat = np.tile(i_bias_vec, (len(train_mat[:, 0]), 1))
baseline_mat = global_mean + u_bias_mat + i_bias_mat
train_mat_norm_simp = train_mat - baseline_mat
train_mat_norm_simp = np.nan_to_num(train_mat_norm_simp)

### Create binary matrix out of appropriate quadrants
ur_mat = np.copy(train_mat)
non_na_ind = np.argwhere(~np.isnan(ur_mat))
ur_mat[non_na_ind[:,0], non_na_ind[:,1]] = 1
ur_mat = np.nan_to_num(ur_mat)
ul_mat = np.zeros((len(train_mat[:,0]), len(train_mat[:,0])))
np.fill_diagonal(ul_mat, np.nansum(ur_mat, axis=1))
lr_mat = np.zeros((len(train_mat[0,:]), len(train_mat[0,:])))
np.fill_diagonal(lr_mat, np.nansum(ur_mat, axis=0))
ll_mat = ur_mat.transpose()
train_mat_bin = np.vstack((np.hstack((ul_mat, ur_mat)), np.hstack((ll_mat, lr_mat))))

### Create means vectors
u_means_vec = np.nanmean(train_mat, axis=1)
i_means_vec = np.nanmean(train_mat, axis=0)
ui_means_vec = np.concatenate([u_means_vec, i_means_vec])
mean_u_means = np.nanmean(u_means_vec)
mean_i_means = np.nanmean(i_means_vec)
u_sums_vec = np.nansum(train_mat, axis=1)
i_sums_vec = np.nansum(train_mat, axis=0)
ui_sums_vec = np.concatenate([u_sums_vec, i_sums_vec])

### Set up solvable matrix version of Bayes / ridge regression by adding to the diagonal of A^T A
K = 3
ATA = np.copy(train_mat_bin)
np.fill_diagonal(ATA, np.diagonal(ATA) + K)
ATb = np.copy(ui_sums_vec)
u_etas = np.repeat(mean_u_means, len(u_means_vec))
i_etas = np.repeat(mean_i_means, len(i_means_vec))
etas = np.concatenate([u_etas, i_etas])
ATb = ATb + etas * K

### Add intercept by adding a row and column to ATA which is total number of ratings (including the added Bayes ratings)
###### concatenated on both sides with number of user (item) ratings (in this case it is diagonal of ATA matrix bc A is
###### all 1s and 0s, so don't need to worry about squared terms.  i.e., 1^2 = 1)
### And add a value to ATb which is the total sum of ratings (including the added Bayes ratings)
total_n_ratings = len(non_na_ind) + len(np.diagonal(ATA)) * K
new_top_row = np.hstack(( total_n_ratings, np.diagonal(ATA) ))
new_side_column = np.diagonal(ATA).reshape(-1,1)
ATA = np.vstack(( new_top_row, np.hstack(( new_side_column, ATA )) ))
total_sum_ratings = np.nansum(u_sums_vec) + np.nansum(etas * K)
ATb = np.concatenate([np.array([total_sum_ratings]), ATb])

### Solve for x
x = np.linalg.solve(ATA, ATb)

### Populate a prediction matrix using the regression coefficients
train_mat_pred = np.full(train_mat.shape, np.nan)
for i in range(len(train_mat_pred[:,0])):
    train_mat_pred[i, ] = x[0] + x[i+1] + x[(len(u_means_vec)+1):len(x)]
below_1_inds = np.argwhere(train_mat_pred<1)
train_mat_pred[below_1_inds[:,0], below_1_inds[:,1]] = 1   #round any value below 1 or above 5
above_5_inds = np.argwhere(train_mat_pred>5)
train_mat_pred[above_5_inds[:,0], above_5_inds[:,1]] = 5
train_mat_norm = train_mat - train_mat_pred
train_mat_norm = np.nan_to_num(train_mat_norm)

### Prediction matrix using standard baseline
train_mat_pred_simp = train_mat_norm_simp + baseline_mat
below_1_inds = np.argwhere(train_mat_pred_simp<1)
train_mat_pred_simp[below_1_inds[:,0], below_1_inds[:,1]] = 1
above_5_inds = np.argwhere(train_mat_pred_simp>5)
train_mat_pred_simp[above_5_inds[:,0], above_5_inds[:,1]] = 5

### Calculate RMSE on Test data
def rmse(predicted, observed, rmna=False):
    if rmna==True:
        return np.nanmean((predicted - observed)**2)**0.5
    else:
        return np.mean((predicted - observed)**2)**0.5

### regression based
rmse_pred = rmse(train_mat_pred, test_mat, rmna=True)
print(rmse_pred)

### simple baseline
rmse_pred_simp = rmse(train_mat_pred_simp, test_mat, rmna=True)
print(rmse_pred_simp)

### Perform SVD on regression baseline
u_regr, s_regr, vt_regr = np.linalg.svd(train_mat_norm)
k = 10
u_comp = u_regr[:, 0:k]
s_comp = s_regr[0:k]
vt_comp = vt_regr[0:k, :]
train_mat_norm_comp = u_comp.dot(np.diag(s_comp)).dot(vt_comp)
train_mat_pred_svd = train_mat_norm_comp + train_mat_pred
below_1_inds = np.argwhere(train_mat_pred_svd<1)
train_mat_pred_svd[below_1_inds[:,0], below_1_inds[:,1]] = 1
above_5_inds = np.argwhere(train_mat_pred_svd>5)
train_mat_pred_svd[above_5_inds[:,0], above_5_inds[:,1]] = 5

### RMSE on test data for regression baseline
rmse_pred_svd = rmse(train_mat_pred_svd, test_mat, rmna=True)
print(rmse_pred_svd)

### Perform SVD on standard baseline
u_simp, s_simp, vt_simp = np.linalg.svd(train_mat_norm_simp)
u_comp = u_simp[:, 0:k]
s_comp = s_simp[0:k]
vt_comp = vt_simp[0:k, :]
train_mat_norm_simp_comp = u_comp.dot(np.diag(s_comp)).dot(vt_comp)
train_mat_pred_simp_svd = train_mat_norm_simp_comp + train_mat_pred_simp
below_1_inds = np.argwhere(train_mat_pred_simp_svd<1)
train_mat_pred_simp_svd[below_1_inds[:,0], below_1_inds[:,1]] = 1
above_5_inds = np.argwhere(train_mat_pred_simp_svd>5)
train_mat_pred_simp_svd[above_5_inds[:,0], above_5_inds[:,1]] = 5

### RMSE on test data for simple baseline
rmse_pred_simp_svd = rmse(train_mat_pred_simp_svd, test_mat, rmna=True)
print(rmse_pred_simp_svd)

### Perform Funk SVD on regression baseline
train_df_norm = pd.DataFrame(data = train_mat_norm).stack().reset_index().rename(columns={'level_0':'u_id', 'level_1':'i_id', 0:'rating'})
train_df_norm.rating = train_df_norm.rating.replace(0, np.nan)   #go back to keeping nan values
f_svd_train = fsvd.SVD(learning_rate=0.001, regularization=0.015, n_epochs=200, n_factors=k)
f_svd_train.fit(train_df_norm)
train_mat_norm_comp = f_svd_train.pu.dot(f_svd_train.qi.transpose())
train_mat_pred_sgd = train_mat_norm_comp + train_mat_pred
below_1_inds = np.argwhere(train_mat_pred_sgd<1)
train_mat_pred_sgd[below_1_inds[:,0], below_1_inds[:,1]] = 1
above_5_inds = np.argwhere(train_mat_pred_sgd>5)
train_mat_pred_sgd[above_5_inds[:,0], above_5_inds[:,1]] = 5

### RMSE on test data for regression baseline
rmse_test_sgd = rmse(train_mat_pred_sgd, test_mat, rmna=True)
print(rmse_test_sgd)

### Perform Funk SVD on standard baseline
train_df_norm_simp = pd.DataFrame(data = train_mat_norm_simp).stack().reset_index().rename(columns={'level_0':'u_id', 'level_1':'i_id', 0:'rating'})
train_df_norm_simp.rating = train_df_norm_simp.rating.replace(0, np.nan)   #go back to keeping nan values
f_svd_train_simp = fsvd.SVD(learning_rate=0.001, regularization=0.015, n_epochs=200, n_factors=k)
f_svd_train_simp.fit(train_df_norm_simp)
train_mat_norm_simp_comp = f_svd_train_simp.pu.dot(f_svd_train_simp.qi.transpose())
train_mat_pred_simp_sgd = train_mat_norm_simp_comp + train_mat_pred_simp
below_1_inds = np.argwhere(train_mat_pred_simp_sgd<1)
train_mat_pred_simp_sgd[below_1_inds[:,0], below_1_inds[:,1]] = 1
above_5_inds = np.argwhere(train_mat_pred_simp_sgd>5)
train_mat_pred_simp_sgd[above_5_inds[:,0], above_5_inds[:,1]] = 5

### RMSE on test data for standard baseline
rmse_test_simp_sgd = rmse(train_mat_pred_simp_sgd, test_mat, rmna=False)
print(rmse_test_simp_sgd)
