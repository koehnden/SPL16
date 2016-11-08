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
#source("Modeling/xgb_nrounds_tune") # to tune nrounds while holding other parameters fixed

# set labels and exclude them from the training set
y <- train$y
train <- train[,-c(1,ncol(train))]

# save result path
result_path <- "Modeling/Results/xgboost/tree_specific/xgb_treespecific_train"

########## perform repeated nested cv
# set cv parameter
t_outer <- 5 # repetitions on the outer loop (not implemented)
k_outer <- 5 # fold of the outer cv loop (not yet implemented)
t_inner <- 5 # repetition on inner loop (not implemented)
k_inner <- 5 # folds on the inner cv loop

# create Grid for GridSearch to tune hyperparameter 
# Tree specific Parameters: maxnodes: longest path of a single tree (decreased performance)
#                           colsample_bytree: variable considered at each split
#                           subsample: size of the bagging bootstrap sample

nrounds_fixed <- 200 # number of trees: tune_nrounds gave ~150 but with a sd of 50 --> I set it to 200 
eta_fixed <- 0.1 # learning rate (fixed for now)
treeSpecificGrid <-  expand.grid(max_depth = seq(4,10,2), 
                                 gamma = seq(0,0.2,0.05), 
                                 subsample = seq(0.6,1,0.2), 
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
tuning_results <- cbind(rep(0,nrow(treeSpecificGrid)),treeSpecificGrid)
colnames(tuning_results) <- c("rmse",parameter_names)


# set up empty matrices to be filled with rmse and best_paramer (here only lambda)


### start nested repeated nested cv loop
## Repetition outer loop
# create empty vector/matrix to save best results
rmse_temp <- matrix(0, nrow = k_inner, ncol = t_inner)
best_parameter <- matrix(0, nrow = k_inner * t_inner, ncol = numOfParameter) # every 10th row represent a new repetition 
colnames(best_parameter) <- parameter_names

# start nested cv loop
for(t_1 in 1:t_inner){
  cat("\n")
  cat("start inner repetition", t_1, "out of", t_inner)
  # draw random integers for the k-folds
  folds_inner <- sample(rep(1:k_inner, length.out = nrow(train)))
  
  # start crossvalidation loop
  for(k_1 in 1:k_inner){
    # split into training and validation set
    indexValidation <- which(folds_inner == k_1)
    training <- train[-indexValidation,]
    y_training <- y[-indexValidation]
    validation <- train[indexValidation,]
    y_validation <- y[indexValidation]
    
    # convert for data into a format xgb.train can handle
    dtrain <- xgb.DMatrix(data = as.matrix(training), label=y_training)
    dvalidation <- xgb.DMatrix(data = as.matrix(validation), label=y_validation)
    # start GridSearch loop 
    for(i in 1:nrow(treeSpecificGrid)){
      # determine arbitrary xgboost parameters in a list
      xgb_paramters = list(                                              
        eta = eta_fixed,                                            # learning rate                                                                
        max.depth = treeSpecificGrid$max_depth[i],                  # max nodes of a tree                                                       
        gamma = treeSpecificGrid$gamma[i],                          # minimal improvement per iteration
        colsample_bytree = treeSpecificGrid$colsample_bytree[i],    # fraction of variable to consider per tree (similar to mtry in rf)
        subsample = treeSpecificGrid$subsample[i],                  # fraction of the whole sample that the bootstrap sample should consist of 
        eval_metric = "rmse",                                      # error metric
        maximize = FALSE
        )
      # fit the randomForrest
      xgbFit <- xgb.train(params = xgb_paramters,  # list of parameter previously specified
                          data =  dtrain,
                          booster = "gbtree",
                          nround = nrounds_fixed,    #number of trees (set accordingly to tune_nrounds function) 
                          verbose = TRUE,
                          objective = "reg:linear"
                          )
      # predict SalePrice
      yhat <- predict(xgbFit, newdata = dvalidation)
      # fill the first column of this matrix with the rmse results (of the log outputs)
      validation_error <- rmse_log(y_validation,yhat)
      tuning_results[i,1] <- validation_error
      # save all training results as csv file (fold_k_reptetion_t)
      write.csv(tuning_results, file = paste(result_path, k_1,t_1,".csv", sep="_"))
      cat("\n")
      cat(validation_error)
    }#end GridSearch
    # find the index of the best model 
    idx_best <- which(tuning_results$rmse == min(tuning_results$rmse))
    # get rmse and the parameter of the best model
    best_results <- tuning_results[idx_best,]
    # save best results
    rmse_temp[k_1,t_1] <- best_results$rmse # best rmse
    best_parameter[k_1*t_1,] <- cbind(best_results$max_depth,
                                      best_results$gamma,
                                      best_results$subsample,
                                      best_results$colsample_bytree
    ) 
    # print temporary results
    cat("\n")
    cat("Best RMSE:",  best_results$rmse,
        "eta:",   best_results$eta,
        "nrounds:",  best_results$nrounds,
        "max_depth:", best_results$max_depth,
        "inner fold:", k_1, "out of", k_inner
    )
  }#end inner cv
}#end inner repetition

# get average prediction error per repetition
colMeans(rmse_temp)
table(best_parameter[,1])
table(best_parameter[,2])
table(best_parameter[,3])
table(best_parameter[,4])