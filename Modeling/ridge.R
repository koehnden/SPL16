################################# Ridge Regression ######################################################################
# This script performs repeated nested cross-validation to tune and estimate the performance of Ridge Regression. 
# Best parameter (here just lambda) is between 0.035 and 0.04
library(caret)

###########################Load Data Cleaning Scripts
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")  # to get performance metrics 
# get preprocessed data
train <- naive_preprocessing(X_com,y)

# save result path (change according to experiment here: input date is from quick preprocessing function)
result_path <- "Modeling/Results/ridge/quick_preprocessing"
# set cv parameter
t_outer <- 5 # repetitions on the outer loop
k_outer <- 10 # fold of the outer cv loop
t_inner <- 10 # repetition on inner loop (here caret does it)
k_inner <- 5 # folds on the inner cv loop

# create Grid for GridSearch to tune hyperparameter (here just lambda)
ridgeGrid <-  expand.grid(lambda = seq(0.04,0.04,0.005)) 

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
  cat("outer repetition", t_1, "out of", t_outer)
  # start k fold outer loop
  for(k_1 in 1:k_outer){
    # random split of the data in k_outer folds
    indexValidation <- which(folds_outer == k_1)
    training <- train[-indexValidation,]
    validation <- train[indexValidation,]
    
    # determine model
    ridgeFit <- train(y ~., 
                      data = training, # exclude Id Variable from training data
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
    cat("\n")
    cat("best parameter:", best_parameter[k_1,t_1], "RMSE:", rmse_temp[k_1,t_1], 
        "outer fold:", k_1, "out of", k_outer)
  }#end k_outer
}#end t_outer

# get average prediction error and sd
rmse_mean <- colMeans(rmse_temp, na.rm = T); rmse_mean 
rmse_sd <- apply(rmse_temp,2,sd); mean(rmse_sd)
# show best parameter choosen by the inner repeated cv 
table(best_parameter)