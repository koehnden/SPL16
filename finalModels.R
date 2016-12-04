################### Run Final Models #######################################################

############################## Ridge Regression ##########################################################3
final_training_ridge <- function(train, test, lambda=0.03){
  # create Grid 
  ridgeGrid <-  expand.grid(lambda = lambda) 
  # determine evaluation method of the cv loop 
  ctrl <- trainControl(method = "none",
                       savePredictions = TRUE
  )
  print("start training final training...")
  # determine model for data set 1
  ridgeFit <- train(y ~., 
                        data = train, # exclude Id Variable from training data
                        method = 'ridge',  # method
                        trControl = ctrl,  # evaluatio method (repeated CV)
                        tuneGrid = ridgeGrid, # grid
                        selectionFunction = oneSE, # oneSE to choose simplest model in condifidence intervall (best alternative) 
                        metric = "RMSE"  # error metric
                        # verbose = True # print steps
  )
  # predict label for the test set using the training parameters
  yhat <- predict(ridgeFit, test)
  return(yhat)
}

## apply ridge regression
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
yhat <- final_training_ridge(train,test,0.03)
yhat[yhat < 0] <- quantile(y,0.95) # one prediction is negative 
write.csv(yhat,file = "Modeling/Results/finalSubmission/ridge.csv")
# quite bad score 0.29656

####################################### xgboost ##################################################

final_training_xgb <- function(train, y, test, xgb_paramters, nrounds_fixed = 1000){
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


###### training with top 30 features 
source("load_ames_data.R")
X <- read.csv("Data/top_30_complete.csv")[,-1] 
train <- X[1:1460,]
test <- X[1461:2919,]
# determine arbitrary xgboost parameters in a list
xgb_paramters = list(                                              
  eta = 0.025,                                  # learning rate                                                                
  max.depth = 16,                  # max nodes of a tree                                                       
  gamma = 0,                          # minimal improvement per iteration
  colsample_bytree = 0.8,    # fraction of variable to consider per tree (similar to mtry in rf)
  subsample = 0.6,                  # fraction of the whole sample that the bootstrap sample should consist of 
  eval_metric = "rmse",                                       # error metric
  maximize = FALSE
)
yhat <- final_training_xgb(train, y, test, xgb_paramters)
write.csv(yhat,"Modeling/Results/finalSubmission/xgb_top_30.csv", row.names = FALSE)
# Kaggle score: 0.14364

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
train_h2o <- as.h2o(training)
validation_h2o <- as.h2o(validation)

rfFit <- h2o.randomForest(           # h2o.randomForest function
  training_frame = train_h2o,        # H2O frame for training
  validation_frame = validation_h2o, # H2O frame for validation (not required)
  x=col_input,                       # predictor columns, by column index
  y=col_label,                       # label column index
  model_id = "rf_covType_v1",        # name the model in H2O
  ntrees = 200,                      # number of trees 
  mtries = 10,                       # number of variable considered at each split
  sample_rate = 0.6,                 # fraction of booststrap sample
  nfolds = 3,
  stopping_metric = "MSE"
)               


###################### SVM with Gaussian Kernel ##########################################################
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

## apply ridge regression
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


write.csv(yhat,file = "Modeling/Results/finalSubmission/gaussianSVM.csv",row.names = FALSE)
# best score so far 0.13..






