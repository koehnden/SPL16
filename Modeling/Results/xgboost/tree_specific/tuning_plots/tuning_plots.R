################################ visualize tuning results ############################################################
library(ggplot2)
library(Rmisc)
library(data.table)

# load visualization function
source("utils/visulizations.R")

# load all inner tuning results
load_data <- function(wd="~/SPL16/Modeling/Results/xgboost/tree_specific/vi_top_30") { 
  setwd(wd)
  files <- list.files()
  tables <- lapply(files, read.csv)
  df <- do.call(rbind, tables)
  return(df)
}
results <- load_data()

# plot result per parameter
max_depth <- box_hyperparameter(results, results$max_depth,"Max Depth") 
gamma <- box_hyperparameter(results, results$gamma,"Gamma")
subsample <- box_hyperparameter(results, results$subsample,"Subsample")
colbytree <- box_hyperparameter(results, results$colsample_bytree,"Column by Tree")
multiplot(max_depth, gamma, subsample, colbytree, cols=2)

p1 <- hyperparameter_heatmap(results, results$max_depth, results$gamma, "Max Depth", "Gamma") 
p2 <- hyperparameter_heatmap(results, results$subsample, results$colsample_bytree, "Bootstrap Sample", "Variables by Tree")
p3 <- hyperparameter_heatmap(results, results$max_depth, results$colsample_bytree, "Max Depth", "Variables by Tree")
p4 <- hyperparameter_heatmap(results, results$max_depth, results$subsample, "Max Depth", "Subsample")
p5 <- hyperparameter_heatmap(results, results$gamma, results$colsample_bytree, "Gamma", "Variables by Tree")
p6 <- hyperparameter_heatmap(results, results$gamma, results$subsample, "Gamma", "Booststrap Sample")
multiplot(p1,p2,p3,p4,p5,p6,cols = 3)

# further inspection of tuning results
hist(results$rms, breaks = "Scott")
dt <- data.table(results)
avg_rmse <- dt[,.(avg=mean(rmse),med=median(rmse),std=sd(rmse)),X]
smallest_rmse <- avg_rmse[med < quantile(med,0.1),X]
avg_rmse[avg == min(avg),X]
avg_rmse[std == min(std),.(X,std)]
# get the top ten percent parameter sets
result_path <- "Modeling/Results/xgboost/tree_specific/tuning_plots/"
all_avg_results <- cbind(avg_rmse, results[1:216,-c(1,2)])
all_avg_results <- all_avg_results[with(all_avg_results, order(avg)), ]
write.csv(all_avg_results,paste(result_path,'sortedAvgResults.csv', sep = ""))


top_ten_results <- all_avg_results[1:10,] 
write.csv(top_ten_results,paste(result_path,'top_ten_results.csv', sep = ""))

