library(caret) 
library(xgboost)
library(Matrix)
library(FSelector)
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")  # to get performance metrics 
# get preprocessed data
train <- basic_preprocessing(X_com,y)$train
# set labels and exclude them from the training set
y <- train$y
train$y <- NULL

########## perform repeated nested cv
# set cv parameter
k_folds <- 5   # folds in the cv loop
# draw random integers for the k-folds
folds <- sample(rep(1:k_folds, length.out = nrow(train)))
idx_least_important <- NULL
stopping = FALSE
# start crossvalidation loop
while(stopping == FALSE){
  # create empty vector/matrix to save validation error
  validation_error <- rep(0,k_folds)
  if (idx_least_important == NULL){
    train_subset <- train
  } else {
    train_subset <- train[,-idx_least_important]
  }
  for(k in 1:k_folds){
  # split into training and validation set
  indexValidation <- which(folds ==k)
  training <- train_subset[-indexValidation,]
  y_training <- y[-indexValidation]
  validation <- train_subset[indexValidation,]
  y_validation <- y[indexValidation]
  
  # convert for data into a format xgb.train can handle
  dtrain <- xgb.DMatrix(data = as.matrix(training), label=y_training)
  dvalidation <- xgb.DMatrix(data = as.matrix(validation), label=y_validation)
  watchlist <- list(eval = dvalidation, train = dtrain)
    # determine arbitrary xgboost parameters in a list
    xgb_paramters = list(                                              
      eta = 0.025,                                  # learning rate                                                                
      max.depth = 8,                  # max nodes of a tree                                                       
      gamma = 2,                          # minimal improvement per iteration
      colsample_bytree = 0.8,    # fraction of variable to consider per tree (similar to mtry in rf)
      subsample = 0.4,                  # fraction of the whole sample that the bootstrap sample should consist of 
      eval_metric = "rmse",                                       # error metric
      maximize = FALSE
    )
    # fit the xgboost
    xgbFit <- xgb.train(params = xgb_paramters,  # list of parameter previously specified
                        data =  dtrain,
                        booster = "gbtree",
                        nround = 1000,    
                        verbose = 1,
                        objective = "reg:linear",
                        watchlist = watchlist,
                        early.stop.round = 50
    )
    
    # predict SalePrice
    yhat <- predict(xgbFit, newdata = dvalidation)
    # fill the first column of this matrix with the rmse results (of the log outputs)
    validation_error[k] <- rmse_log(y_validation, yhat)
    #get variable importance
    importance_matrix <- xgb.importance(colnames(training), model = xgbFit)
    # get the least important variable and delete it from the training set
    least_important <- importance_matrix[nrow(importance_matrix),]$Feature
    idx_least_important <- which(names(training) %in% least_important)
  }#end cv
  validation_error <- mean(validation_error)
  stopping <- validation_error > previous_validation_error
  previous_validation <- validation_error
}#end fs while
colnames(training)