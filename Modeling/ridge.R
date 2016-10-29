######################## Ridge Regression ########################################
# This script performs repeated nested cross-validation to tune and estimate the performance of
# Ridge Regression. 
library(caret)
#library(doMC)
#registerDoMC(cores = 4)  # only available in linux

# run Data Cleaning Scripts
setwd("~/SPL16")
source("~/SPL16/Data_Cleaning/convert_categoricals.R")
source("~/SPL16/Feature_Selection/delete_nearzero_variables.R") # Put ouT X_com as the cleaned Feature Matrix
source("~/SPL16/utils/performanceMetrics.R") # get performance Metric functions

# remerge train data 
train <- cbind(X_com[1:length(y),],y)

########## perform repeated nested cv
# set cv parameter
t_outer <- 5 # repetitions on the outer loop
k_outer <- 5 # fold of the outer cv loop
t_inner <- 5 # repetition on inner loop (here caret does it)
k_inner <- 5 # folds on the inner cv loop
outer_iteration <- t_outer * k_outer 

# create Grid for GridSearch to tune hyperparameter (here just lambda)
ridgeGrid <-  expand.grid(lambda = seq(0.025,0.075,0.005)) 

# determine evaluation method of the inner cv loop
ctrl <- trainControl(method = "repeatedcv",
                     number = k_inner, # how many folds (k)
                     repeats = t_inner  # how many repitions (t)
                     #allowParallel=TRUE 
)

# set up empty matrices to be filled with rmse and best_paramer (here only lambda)
rmse_temp <- matrix(0, nrow = k_outer, ncol = t_outer)
best_parameter <- matrix(0, nrow = k_outer, ncol = t_outer) 

### start nested repeated nested cv loop
## Repetition outer loop
for(t_1 in 1:t_outer){
  folds_outer <- sample(rep(1:k_outer, length.out = nrow(train)))
  cat("\n")
  cat("outer repetition", t_1, "out of", t_inner)
  # start k fold outer loop
  for(k_1 in 1:k_outer){
    # random split of the data in k_outer folds
    indexValidation <- which(folds_outer == k_1)
    training <- train[-indexValidation,]
    validation <- train[indexValidation,]
    
    # determine model
    ridgeFit <- train(y ~., 
                      data = training[,-1], # exclude Id Variable from training data
                      method = 'ridge',  # method
                      trControl = ctrl,  # evaluatio method (repeated CV)
                      tuneGrid = ridgeGrid, # grid
                      selectionFunction = oneSE, # oneSE to choose simplest model in condifidence intervall (best alternative) 
                      metric = "RMSE"  # error metric
                      # verbose = True # print steps
    )
    # get the best set of parameters (here just lambda) 
    best_parameter[k_1,t_1] <- as.numeric(ridgeFit$bestTune)
    ## predict using the best model (caret does this automatically)
    yhat <- predict(ridgeFit, newdata = validation)
    # compute inner fold rmse
    rmse_temp[k_1,t_1] <- rmse_log(validation$y,yhat)
    # print temporary results
    outer_iteration <- outer_iteration - 1
    cat("\n")
    cat("best parameter:", best_parameter[k_1,t_1], "RMSE:", rmse_temp[k_1,t_1], 
        "outer fold:", k_1, "out of", k_outer)
  }#end k_outer
}#end t_outer

# get average prediction error and sd
rmse_mean <- colMeans(rmse_temp); mean(rmse_mean)
rmse_sd <- apply(rmse_temp,2,sd); mean(rmse_sd)
mean(rmse_mean) + 1.96 * mean(rmse_sd)
# show best parameter choosen by the inner repeated cv 
table(best_parameter)
