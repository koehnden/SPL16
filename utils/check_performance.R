###################################### Test prepocessing ##################################################################
# This script provides functions to run a model with two different preprocessing strategies again each other
library(caret)
library(Matrix)
library(xgboost)

# function to test two different preprocession stategies against each other using ridge regression  
performance_test_ridge <- function(train_benchmark, train_new, lambda=0.03, k=5, t=10){
  # set seed to get same fold for both experiments
  seeds <- seq(1,k*t+1)
  # create Grid 
  ridgeGrid <-  expand.grid(lambda = lambda) 
  # determine evaluation method of the cv loop 
  ctrl <- trainControl(method = "repeatedcv",
                       number = k, # how many folds (k)
                       repeats = t,  # how many repetions (t)
                       savePredictions = TRUE,
                       seeds = seeds
  )
  print("start training with the new data...")
  # determine model for data set 1
  ridgeFit_new <- train(y ~., 
                    data = train_new, # exclude Id Variable from training data
                    method = 'ridge',  # method
                    trControl = ctrl,  # evaluatio method (repeated CV)
                    tuneGrid = ridgeGrid, # grid
                    selectionFunction = oneSE, # oneSE to choose simplest model in condifidence intervall (best alternative) 
                    metric = "RMSE"  # error metric
                    # verbose = True # print steps
  )
  print(ridgeFit_new$results)
  print("start training with benchmark data...")
  # determine model for data set 2
  ridgeFit_benchmark <- train(y ~., 
                     data = train_benchmark, # exclude Id Variable from training data
                     method = 'ridge',  # method
                     trControl = ctrl,  # evaluatio method (repeated CV)
                     tuneGrid = ridgeGrid, # grid
                     selectionFunction = oneSE, # oneSE to choose simplest model in condifidence intervall (best alternative) 
                     metric = "RMSE"  # error metric
                     # verbose = True # print steps
  )
  print(ridgeFit_new$results)
  # return results of both data sets
  return(list(ridgeFit_new$results,ridgeFit_benchmark$results))
}
# usage
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")  # to get performance metrics 
# get preprocessed data
train_new <- basic_preprocessing(X_com,y,scaler = "gaussian")$train
train_benchmark <- basic_preprocessing(X_com,y,scaler = "min_max")$train
check_encoding <- performance_test_ridge(train_new=train_new ,train_benchmark=train_benchmark)
# encoding is not significantly better


# function to perform gridSearch and crossvalidation on the xgboost model
xgb_train <- function(train, y, xgbGrid,eta_fixed=0.025, nrounds_fixed=1000, k=5, seeds = 123){
  numOfParameter <- ncol(xgbGrid)  
  # create a matrix to with the GridSearch parameters and their RMSE
  parameter_names <- c("max_depth",
                       "gamma",
                       "subsample",
                       "colsample_bytree"
  )
  # create empty vector/matrix to save results
  tuning_results <- cbind(rep(0,nrow(xgbGrid)),xgbGrid)
  colnames(tuning_results) <- c("rmse",parameter_names)
  # draw random integers for the k-folds
  set.seed(seeds)
  folds <- sample(rep(1:k, length.out = nrow(train)))
  # start crossvalidation loop
  for(fold in 1:k){
    # split into training and validation set
    indexValidation <- which(folds == fold)
    training <- train[-indexValidation,]
    y_training <- as.numeric(y[-indexValidation])
    validation <- train[indexValidation,]
    y_validation <- y[indexValidation]
    
    # convert for data into a format xgb.train can handle
    dtrain <- xgb.DMatrix(data = sapply(training, as.numeric), label=y_training)
    dvalidation <- xgb.DMatrix(data = sapply(validation, as.numeric), label=y_validation)
    watchlist <- list(eval = dvalidation, train = dtrain)
    # start GridSearch loop 
    for(i in 1:nrow(xgbGrid)){
      # determine arbitrary xgboost parameters in a list
      xgb_paramters = list(                                              
        eta = eta_fixed,                                  # learning rate                                                                
        max.depth = xgbGrid$max_depth[i],                  # max nodes of a tree                                                       
        gamma = xgbGrid$gamma[i],                          # minimal improvement per iteration
        colsample_bytree = xgbGrid$colsample_bytree[i],    # fraction of variable to consider per tree (similar to mtry in rf)
        subsample = xgbGrid$subsample[i],                  # fraction of the whole sample that the bootstrap sample should consist of 
        eval_metric = "rmse",                                       # error metric
        maximize = FALSE
      )
      # fit the xgboost
      xgbFit <- xgb.train(params = xgb_paramters,  # list of parameter previously specified
                          data =  dtrain,
                          booster = "gbtree",
                          nround = nrounds_fixed,    #number of trees (set accordingly to tune_nrounds function) 
                          verbose = 1,
                          early.stop.round = 25,
                          objective = "reg:linear",
                          watchlist = watchlist
      )
      # predict SalePrice
      yhat <- predict(xgbFit, newdata = dvalidation)
      # fill the first column of this matrix with the rmse results (of the log outputs)
      validation_error <- rmse_log(y_validation, yhat)
      tuning_results[i,1] <- validation_error
    }#end GridSearch
  }#end inner cv
  return(tuning_results)
}

# function to test two different preprocession stategies against each other using xgboost
performance_test_xgb <- function(train_benchmark, train_new, y, xgbGrid, eta_fixed=0.025, nrounds_fixed=1000, k=5, seeds = 123){
  print("start training with the new data...")
  result_new <- xgb_train(train=train_new,y=y,xgbGrid=xgbGrid)
  print("start training with the benchmark data...")
  result_benchmark <- xgb_train(train=train_benchmark,y=y,xgbGrid=xgbGrid)
  return(list(new_model = result_new, benchmark_model = result_benchmark))
}

# usage
# get preprocessed data to run against each other
train_benchmark <- basic_preprocessing(X_com,y, scaler="min_max")$train
train_new <- basic_preprocessing(X_com,y, scaler = "gaussian")$train
# get labels
y <- train_benchmark$y
train_new$y <- NULL
train_benchmark$y <- NULL
# specify grid 
xgbGrid <-  expand.grid(max_depth = seq(4,10,2), 
                        gamma = 0, # gamma seems to be not be crucial (we do not tune it)
                        subsample = seq(0.6,0.6,0.2), 
                        colsample_bytree = seq(0.85,0.85,0.15)
)
# run both data version against each other
new_vs_bench <- performance_test_xgb(train_new=train_new,
                     train_benchmark=train_benchmark,
                     y=y, 
                     xgbGrid=xgbGrid
                     )