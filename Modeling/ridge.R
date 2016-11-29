################################# Ridge Regression ######################################################################
# This script performs repeated nested cross-validation to tune and estimate the performance of Ridge Regression. 
# Best parameter (here just lambda) is between 0.035 and 0.04
library(caret)

###########################Load Data Cleaning Scripts
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")  # to get performance metrics 
# get preprocessed data
train <- basic_preprocessing(X_com,y)$train
# save result path (change according to experiment here: input date is from quick preprocessing function)
result_path <- "Modeling/Results/ridge/basic_preprocessing/"
# set cv parameter
t_outer <- 5 # repetitions on the outer loop
k_outer <- 10 # fold of the outer cv loop
t_inner <- 5 # repetition on inner loop (here caret does it)
k_inner <- 5 # folds on the inner cv loop

# create Grid for GridSearch to tune hyperparameter (here just lambda)
ridgeGrid <-  expand.grid(lambda = seq(0.01,1,0.05 )) 

# determine evaluation method of the inner cv loop
ctrl <- trainControl(method = "repeatedcv",
                     number = k_inner, # how many folds (k)
                     repeats = t_inner  # how many repitions (t)
                     #allowParallel=TRUE 
)

# set up empty matrices to be filled with rmse and best_paramer (here only lambda)
results <- matrix(0, nrow = k_outer, ncol = 2) 
colnames(results) <- c("rmse","lambda")
result_list <- lapply(seq_len(t_outer), function(X) results)

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
                      data = training, #
                      method = 'ridge',  # method
                      trControl = ctrl,  # evaluatio method (repeated CV)
                      tuneGrid = ridgeGrid, # grid
                      selectionFunction = oneSE, # oneSE to choose simplest model in condifidence intervall (best alternative) 
                      metric = "RMSE"  # error metric
                      # verbose = True # print steps
    )
    # get the best set of parameters (here just lambda) 
    result_list[[t_1]][k_1,2] <- as.numeric(ridgeFit$bestTune)
    ## predict using the best model (caret does this automatically)
    yhat <- predict(ridgeFit, newdata = validation)
    # compute inner fold rmse and save it in the result list
    result_list[[t_1]][k_1,1] <- rmse_log(validation$y,yhat)
    # print temporary results
    cat("\n")
    cat("best parameter:", result_list[[t_1]][k_1,2], "RMSE:", result_list[[t_1]][k_1,1], 
        "outer fold:", k_1, "out of", k_outer)
  }#end k_outer
}#end t_outer
print(result_list)
# save results of the outer loop
for(i in 1:length(result_list)){
  # save all training results as csv file (fold_k_reptetion_t)
  write.csv(result_list[[i]], file = paste(result_path,i,"bestModels.csv", sep="_"))
}