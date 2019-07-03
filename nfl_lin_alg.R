library(plyr)
library(dplyr)
library(rvest)
library(tidytext)

### Scrape data
full_url <- "https://www.pro-football-reference.com/years/2018/games.htm"
url_html <- full_url %>% read_html()
table_list <- url_html %>% html_table(fill=T)
games_df <- table_list[[1]]

### Tidy data
names(games_df)[names(games_df)==""] <- c("h_r", "box")
games_df_filt <- games_df %>% 
  filter(box!="", Week %in% as.character(1:17)) %>% 
  select(W=`Winner/tie`, L=`Loser/tie`, PtsW, PtsL) %>% 
  mutate(pts_diff=as.integer(PtsW)-as.integer(PtsL))
h2h_df <- rbind.fill(games_df_filt, 
                            data.frame(W=games_df_filt$L, 
                                       L=games_df_filt$W, 
                                       PtsW=games_df_filt$PtsL, 
                                       PtsL=games_df_filt$PtsW, 
                                       pts_diff=games_df_filt$pts_diff*-1)) %>% mutate(games = 1)

### Create head-to-head matrix and point differential vector
h2h_mat <- h2h_df %>% group_by(W, L) %>% summarise(g = sum(games)) %>% ungroup() %>% cast_sparse(W, L, g) %>% as.matrix()
h2h_mat <- h2h_mat[,order(colnames(h2h_mat))]
h2h_mat <- h2h_mat * -1
diag(h2h_mat) <- 16
pts_diff_df <- h2h_df %>% group_by(W) %>% summarise(pts_diff = sum(pts_diff)) %>% ungroup()

### Correct for singularity and solve Ax = b
A <- h2h_mat
diag(A) <- diag(A) + 1
b <- pts_diff_df$pts_diff
x <- solve(A, b)

### View new rankings
rank_df <- tibble(team=rownames(A), 
                  b=b, 
                  orig_rank=rank(-b, ties.method = "min"), 
                  x=x*16, 
                  mod_rank=rank(-x, ties.method = "min"))
View(rank_df)
