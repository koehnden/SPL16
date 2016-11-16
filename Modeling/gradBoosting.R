############################# Stochastic Gradient Boosting training using xgboost  #############################################################
# This script tunes the tree specific parameter of a Stoachastic Gradient Boosting Machine using GridSearch and repeated cross-validation 
# It uses the nrounds found by the tune_nrounds() function
# Depending on the parameter set this takes quite a while
# Results of the training are save in csv files under Modeling/Results/xgbosst/xgb_treespecific_train_fold_repetition
# For a quick overview on the xgb parametets: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# or the xgboost documentation on cran
library(caret) 
library(xgboost)
library(Matrix)
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")  # to get performance metrics 

# set labels and exclude them from the training set
y <- train$y
train <- train[,-c(1,ncol(train))]

# save result path (change according to experiment here: input date is from quick preprocessing function)
result_path <- "Modeling/Results/xgboost/tree_specific/quick_preprocessing/xgb_treespecific_train"

########## perform repeated nested cv
# set cv parameter
k_outer <- 10  # fold of the outer cv loop 
k_inner <- 5   # folds on the inner cv loop

# create Grid for GridSearch to tune hyperparameter 
# Tree specific Parameters: maxnodes: longest path of a single tree (decreased performance)
#                           colsample_bytree: variable considered at each split
#                           subsample: size of the bagging bootstrap sample

nrounds_fixed <- 1000 # number of trees: no need for tuning since early.stopping is possible 
eta_fixed <- 0.025 # learning rate (fixed for now)
treeSpecificGrid <-  expand.grid(max_depth = seq(4,10,2), 
                                 gamma = 0, # gamma seems to be not be crucial (we do not tune it)
                                 subsample = seq(0.2,0.8,0.2), 
                                 colsample_bytree = seq(0.1,0.85,0.15)
                                 )
# samplesize could be inspected as well
numOfParameter <- ncol(treeSpecificGrid)  
# create a matrix to with the GridSearch parameters and their RMSE
parameter_names <- c("max_depth",
                     "gamma",
                     "subsample",
                     "colsample_bytree"
)
# set up empty matrices to be filled with rmse and best_paramer (here only lambda)


### start nested repeated nested cv loop
## Repetition outer loop
# create empty vector/matrix to save best results
rmse_temp <- matrix(0, nrow = k_inner, ncol = k_outer)
kbest <- 5 # how many models to be run in the outer loop
best_parameter <- matrix(0, nrow = k_inner, ncol = numOfParameter + 1) 
colnames(best_parameter) <- c("rmse",parameter_names)
result_list <- lapply(seq_len(k_outer), function(X) best_parameter)

# start nested cv loop
for(k_1 in 1:k_outer){
  # draw random integers for the k-folds
  folds_outer <- sample(rep(1:k_outer, length.out = nrow(train)))
  # split into training and validation set
  indexValidation_outer <- which(folds_outer == k_1)
  training_outer <- train[-indexValidation_outer,]
  y_training_outer <- y[-indexValidation_outer]
  validation_outer <- train[indexValidation_outer,]
  y_validation_outer <- y[indexValidation_outer]
  # convert for data into a format xgb.train can handle for the outer training
  dtrain_outer <- xgb.DMatrix(data = as.matrix(training_outer), label=y_training_outer)
  dvalidation_outer <- xgb.DMatrix(data = as.matrix(validation_outer), label=y_validation_outer)
  watchlist_outer <- list(eval = dvalidation_outer, train = dtrain_inner)
  cat("\n")
  cat("start outer nested cv loop", k_1, "out of", k_outer)
  # draw random integers for the k-folds
  folds_outer <- sample(rep(1:k_outer, length.out = nrow(train)))
  
  # create empty vector/matrix to save best results
  tuning_results <- cbind(rep(0,nrow(treeSpecificGrid)),treeSpecificGrid)
  colnames(tuning_results) <- c("rmse",parameter_names)
  # start crossvalidation loop
  for(k_2 in 1:k_inner){
    # draw random integers for the k-folds
    folds_inner <- sample(rep(1:k_inner, length.out = nrow(training_outer)))
    # split into training and validation set
    indexValidation_inner <- which(folds_inner == k_2)
    training_inner <- training_outer[-indexValidation_inner,]
    y_training_inner <- y_training_outer[-indexValidation_inner]
    validation_inner <- training_outer[indexValidation_inner,]
    y_validation_inner <- y_training_outer[indexValidation_inner]
    
    # convert for data into a format xgb.train can handle
    dtrain_inner <- xgb.DMatrix(data = as.matrix(training_inner), label=y_training_inner)
    dvalidation_inner <- xgb.DMatrix(data = as.matrix(validation_inner), label=y_validation_inner)
    watchlist_inner <- list(eval = dvalidation_inner, train = dtrain_inner)
    # start GridSearch loop 
    for(i in 1:nrow(treeSpecificGrid)){
      # determine arbitrary xgboost parameters in a list
      xgb_paramters = list(                                              
        eta = eta_fixed,                                            # learning rate                                                                
        max.depth = treeSpecificGrid$max_depth[i],                  # max nodes of a tree                                                       
        gamma = treeSpecificGrid$gamma[i],                          # minimal improvement per iteration
        colsample_bytree = treeSpecificGrid$colsample_bytree[i],    # fraction of variable to consider per tree (similar to mtry in rf)
        subsample = treeSpecificGrid$subsample[i],                  # fraction of the whole sample that the bootstrap sample should consist of 
        eval_metric = "rmse",                                       # error metric
        maximize = FALSE
        )
      # fit the xgboost
      xgbFit_inner <- xgb.train(params = xgb_paramters,  # list of parameter previously specified
                          data =  dtrain_inner,
                          booster = "gbtree",
                          nround = nrounds_fixed,    #number of trees (set accordingly to tune_nrounds function) 
                          verbose = 1,
                          early.stop.round = 25,
                          objective = "reg:linear",
                          watchlist = watchlist_inner
                          )
      # predict SalePrice
      yhat_inner <- predict(xgbFit_inner, newdata = dvalidation_inner)
      # fill the first column of this matrix with the rmse results (of the log outputs)
      validation_error_inner <- rmse_log(y_validation_inner, yhat_inner)
      tuning_results[i,1] <- validation_error_inner
      # save all training results as csv file (fold_k_reptetion_t)
      write.csv(tuning_results, file = paste(result_path, k_2,k_1,".csv", sep="_"))
    }#end GridSearch
    # find the index of the 5 best models with the smallest rmse 
    idx_kbest <- order(tuning_results$rmse)[1:5]
    # get rmse and the parameter of the best model
    best_results <- tuning_results[idx_kbest,]
    # save best results
    rmse_temp[k_2,k_1] <- best_results$rmse[1] # best rmse
  }#end inner cv
  # fill result list with the best parameter 
  result_list[[k_1]] <- best_results
  # start training with k-best parameter set using the outer training data
  for(model in 1:nrow(best_results)){
    # set xgboost parameters to the k best parameter set
    xgb_paramters = list(                                              
      eta = eta_fixed,                                            # learning rate                                                                
      max.depth = best_results$max_depth[model],                  # max nodes of a tree                                                       
      gamma = best_results$gamma[model],                          # minimal improvement per iteration
      colsample_bytree = best_results$colsample_bytree[model],    # fraction of variable to consider per tree (similar to mtry in rf)
      subsample = best_results$subsample[model],                  # fraction of the whole sample that the bootstrap sample should consist of 
      eval_metric = "rmse",                                       # error metric
      maximize = FALSE
    )
    # fit the xgboost for outer training
    xgbFit_outer <- xgb.train(params = xgb_paramters,    # list of parameter previously specified
                              data =  dtrain_outer,
                              booster = "gbtree",
                              nround = nrounds_fixed,    # number of trees (set accordingly to tune_nrounds function) 
                              verbose = 1,
                              early.stop.round = 25,     
                              objective = "reg:linear",
                              watchlist = watchlist_outer
    )
    # predict SalePrice
    yhat_outer <- predict(xgbFit_outer, newdata = dvalidation_outer)
    # fill the first column of this matrix with the rmse results (of the log outputs)
    validation_error_outer <- rmse_log(y_validation_outer, yhat_outer)
    result_list[[k_1]][model,1] <- validation_error_outer
    # save all training results as csv file (fold_k_reptetion_t)
    write.csv(result_list[[k_1]], file = paste(result_path,k_2,model,"bestModels.csv", sep="_"))
  }# end outer training
}#end outer cv loop
# print result list
print(result_list) 