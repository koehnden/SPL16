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

# get preprocessed data
train <- basic_preprocessing(X_com,y)$train
test <-  basic_preprocessing(X_com,y)$test
X_com <- rbind(train,test)


# set labels and exclude them from the training set
y <- train$y
train$y <- NULL

# save result path (change according to experiment here: input date is from quick preprocessing function)
result_path <- "Modeling/Results/xgboost/tree_specific/basic_preprocessing/xgb_basic"

########## perform repeated nested cv
# set cv parameter
repetitions <- 5  # repetitions of cv 
k_folds <- 5   # folds in the cv loop

# create Grid for GridSearch to tune hyperparameter 
# Tree specific Parameters: maxnodes: longest path of a single tree (decreased performance)
#                           colsample_bytree: variable considered at each split
#                           subsample: size of the bagging bootstrap sample

nrounds_fixed <- 1000 # number of trees: no need for tuning since early.stopping is possible 
eta_fixed <- 0.025 # learning rate (fixed for now)
treeSpecificGrid <-  expand.grid(max_depth = seq(10,16,2), 
                                 gamma = seq(0,6,2), # gamma seems to be not be crucial (we do not tune it)
                                 subsample = seq(0.4,0.8,0.2), 
                                 colsample_bytree = seq(0.6,1,0.2)
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
parameters <- matrix(0, nrow = nrow(treeSpecificGrid), ncol = numOfParameter + 1) 
colnames(parameters) <- c("rmse",parameter_names)
result_list <- lapply(seq_len(repetitions), function(X) parameters)

# start repetition loop
for(t in 1:repetitions){
  # draw random integers for the k-folds
  folds <- sample(rep(1:k_folds, length.out = nrow(train)))
  # create empty vector/matrix to save best results
  tuning_results <- cbind(rep(0,nrow(treeSpecificGrid)),treeSpecificGrid)
  colnames(tuning_results) <- c("rmse",parameter_names)
  # start crossvalidation loop
  for(k in 1:k_folds){
    # split into training and validation set
    indexValidation <- which(folds ==k)
    training <- train[-indexValidation,]
    y_training <- y[-indexValidation]
    validation <- train[indexValidation,]
    y_validation <- y[indexValidation]
    
    # convert for data into a format xgb.train can handle
    dtrain <- xgb.DMatrix(data = as.matrix(training), label=y_training)
    dvalidation <- xgb.DMatrix(data = as.matrix(validation), label=y_validation)
    watchlist <- list(eval = dvalidation, train = dtrain)
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
      xgbFit <- xgb.train(params = xgb_paramters,  # list of parameter previously specified
                          data =  dtrain,
                          booster = "gbtree",
                          nround = nrounds_fixed,    #number of trees (set accordingly to tune_nrounds function) 
                          verbose = 1,
                          early.stop.round = 50,
                          objective = "reg:linear",
                          watchlist = watchlist
      )
      # predict SalePrice
      yhat <- predict(xgbFit, newdata = dvalidation)
      # fill the first column of this matrix with the rmse results (of the log outputs)
      validation_error <- rmse_log(y_validation, yhat)
      tuning_results[i,1] <- validation_error
      # save all training results as csv file (fold_k_reptetion_t)
      write.csv(tuning_results, file = paste(result_path,t,k,".csv", sep="_"))
    }#end GridSearch
    #rmse_temp[k_2,k_1] <- best_results$rmse[1] # best rmse
  }#end inner cv
  # fill result list with the best parameter 
  result_list[[t]] <- tuning_results
}#end repetitions
# print result list
print(result_list) 

results_outer <- rbind(result_list[[1]],result_list[[2]],result_list[[3]],result_list[[4]],result_list[[5]],
                       result_list[[6]],result_list[[7]],result_list[[8]],result_list[[9]],result_list[[10]])
write.csv(results_outer, paste(result_path,"mergeBestResult.csv"))