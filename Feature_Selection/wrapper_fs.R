library(caret) 
library(xgboost)
library(Matrix)
library(FSelector)
# load script to get importance_matrix
source("Feature_Selection/xgb_variable_importance.R") # takes less than 2min
########## perform repeated nested cv
# get the feature ordered by their importance 
importance_ranking <- importance_matrix$Feature

# function to perform wrapper feature selection using beackwards elimination with xgboost
backwards_elimination <- function(train, y, importance_matrix, k_folds = 5, repetitions = 10){
  # get the feature ordered by their importance 
  importance_ranking <- importance_matrix$Feature
  train_subset <- train
  validation_results <- c(rep(0,length(importance_ranking)),100)
  for(i in length(importance_ranking):1){
    # create empty list to save validation results
    validation_median <- c(rep(0,repetitions))
    idx_least_important <- which(names(train_subset) %in% importance_ranking[i])
    train_subset <- train_subset[,-idx_least_important]
    for(t in 1:repetitions){
      # create empty list to save validation results
      validation_error <- c(rep(0,k_folds))
      # draw random integers for the k-folds
      folds <- sample(rep(1:k_folds, length.out = nrow(train)))
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
          eta = 0.025,                    # learning rate                                                                
          max.depth = 8,                  # max nodes of a tree                                                       
          gamma = 2,                      # minimal improvement per iteration
          colsample_bytree = 0.8,         # fraction of variable to consider per tree (similar to mtry in rf)
          subsample = 0.4,                # fraction of the whole sample that the bootstrap sample should consist of 
          eval_metric = "rmse",           # error metric
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
        # fill the validation_error vector with the corresponding rmse
        validation_error[k] <- rmse_log(y_validation, yhat)
      }#end cv loop
      validation_median[t] <- median(validation_error)
    }#end repetition loop
    # calculate the mean rmse
    validation_results[i] <- mean(validation_median)
    # check if results are improved by excluding the variable
    if(validation_results[i]  > validation_results[i+1]){
      # keep variable
      train_subset[important_ranking[i]] <- train[important_ranking[i]]
    }
  }#end backwards elimination loop
  return(list(choosen_subset = train_subset, rmse_scores = validation_results))
}
# usage
X_fs <- backwards_elimination(train,y,importance_matrix)

