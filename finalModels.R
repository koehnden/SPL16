################### Run Final Models ##################################################################
# This script run the final models with the best parameter recovered from the tuning process. 
# Note that recovered tuning parameter are different for the three different training set we use, i.e.
#   - data with all Variables
#   - data with only the top 30 variables 
#   - data with the pca from all the variables as input
library(caret)
library(xgboost)
library(Matrix)
library(h2o)

####################################### xgboost ##################################################
# Function to train the final xgboost model
# Note: We put the best parameter from recovered from xgb_tuning into the model and
# train it on the whole training set and predict unseen test data 
final_training_xgb <- function(train, test, xgb_paramters, nrounds_fixed = 1000){
  # convert for data into a format xgb.train can handle
  dtrain <- xgb.DMatrix(data = sapply(train, as.numeric), label=y)
  # fit the xgboost
  xgbFit <- xgb.train(params = xgb_paramters,  # list of parameter previously specified
                      data =  dtrain,
                      booster = "gbtree",
                      nround = nrounds_fixed,    #number of trees (set accordingly to tune_nrounds function) 
                      verbose = 1,
                      objective = "reg:linear"
  )
  # predict SalePrice
  yhat <- predict(xgbFit, newdata = sapply(test,as.numeric))
  yhat <- cbind(as.numeric(rownames(test)),yhat)
  colnames(yhat) <- c("Id","SalePrice")
  return(yhat)
}

# determine arbitrary xgboost parameters in a list
xgb_paramters = list(                                              
  eta = 0.025,                                  # learning rate                                                                
  max.depth = 16,                  # max nodes of a tree                                                       
  gamma = 0,                          # minimal improvement per iteration
  colsample_bytree = 0.8,    # fraction of variable to consider per tree (similar to mtry in rf)
  subsample = 0.6                  # fraction of the whole sample that the bootstrap sample should consist of 
)
# run the model
yhat <- final_training_xgb(train, test, xgb_paramters)
# save prediction that are submitted to kaggle.com
write.csv(yhat,"Modeling/Results/finalSubmission/xgb_basic.csv", row.names = FALSE)
# Kaggle score: 0.1333

################# PCA(80%) + xgboost #############################3
X <- pca_preprocessing(X_com,y,0.8)
train <- X$train
y <- train$ytrain
train$y <- NULL
test <- X$test
# determine arbitrary xgboost parameters in a list
xgb_paramters = list(                                              
  eta = 0.025,                                  # learning rate                                                                
  max.depth = 12,                  # max nodes of a tree                                                       
  gamma = 0,                          # minimal improvement per iteration
  colsample_bytree = 1,    # fraction of variable to consider per tree (similar to mtry in rf)
  subsample = 0.8,                  # fraction of the whole sample that the bootstrap sample should consist of 
  eval_metric = "rmse",                                       # error metric
  maximize = FALSE
)
yhat <- final_training_xgb(train, y, test, xgb_paramters)
write.csv(yhat,"Modeling/Results/finalSubmission/xgb_pca2.csv", row.names = FALSE)
# Kaggle Score: 0.15364

############################## Random Forest with H2o ####################################
# same procedure as with xgboost, we put the best parameter set recovered from
# rf_tuning into the model and train on the whole training set

# convert into h2o format
train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)
col_label <- which(colnames(train) == "y")
col_input <- which(colnames(train) != "y")

# specify parameters and run model
rfFit <- h2o.randomForest(           # h2o.randomForest function
  training_frame = train_h2o,        # H2O frame for training
  x=col_input,                       # predictor columns, by column index
  y=col_label,                       # label column index
  model_id = "rf_covType_v1",        # name the model in H2O
  ntrees = 300,                      # number of trees 
  mtries = 30,                       # number of variable considered at each split
  sample_rate = 0.8,                 # fraction of booststrap sample
  max_depth = 20
)               
# make predictions
yhat <- h2o.predict(rfFit, newdata = test_h2o)
# convert prediction back into a data.frame and into the format kaggle requires
yhat <- as.data.frame(yhat)
yhat <- cbind(1461:2919,yhat)
colnames(yhat) <- c("Id","SalePrice")
# save as csv file to submit to kaggle
write.csv(yhat,"Modeling/Results/finalSubmission/rf_pca.csv", row.names = FALSE)
# with top 30 features: 0.15154
# with full set: 0.14836
# with pca (80%): 0.16570

###################### SVR with Gaussian Kernel ##########################################################
# function train the final gaussian svr 
final_training_gaussianSVM <- function(train, test, C = 4.5, sigma = 0.002){
  # create Grid 
  svmGaussianGrid <- expand.grid(C = C, sigma = sigma) 
  # determine evaluation method of the cv loop 
  ctrl <- trainControl(method = "none",
                       savePredictions = TRUE
  )
  print("start training final training...")
  # determine model 
  Fit <- train(y ~., 
               data = train, # exclude Id Variable from training data
               method = 'svmRadial',  # method
               trControl = ctrl,  # evaluatio method (repeated CV)
               tuneGrid = svmGaussianGrid, # grid
               metric = "RMSE"  # error metric
                    # verbose = True # print steps
  )
  # predict label for the test set using the training parameters
  yhat <- predict(Fit, test)
  return(yhat)
}

## apply support vector regression
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")
# get preprocessed data
preprocessed_data <- basic_preprocessing(X_com,y)
# train set
train <- preprocessed_data$train
# test set
test <- preprocessed_data$test
# get predictions
yhat <- cbind(1461:2919,final_training_gaussianSVM(train,test))
colnames(yhat) <- c("Id","SalePrice")
# save as a csv file for kaggle
write.csv(yhat,file = "Modeling/Results/finalSubmission/gaussianSVM.csv",row.names = FALSE)
# best score so far 0.13..