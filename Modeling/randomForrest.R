library(randomForest)
library(caret)

########## perform repeated nested cv
# set cv parameter
t_outer <- 5 # repetitions on the outer loop
k_outer <- 5 # fold of the outer cv loop
t_inner <- 5 # repetition on inner loop (here caret does it)
k_inner <- 5 # folds on the inner cv loop

# create Grid for GridSearch to tune hyperparameter (here just lambda)
rfGrid <-  expand.grid(mtry = seq(10,50,10), ntree = c(300,600,900), maxnodes = c(5,10,15,20))  
# samplesize could 
numOfParameter <- ncol(rfGrid)  

# create a matrix to with the GridSearch parameters and their RMSE
tuning_results <- cbind(rep(0,nrow(rfGrid)),rfGrid)
colnames(tuning_results) <- c("rmse","mtry","ntree","max_nodes")

# create empty vector/matrix to save best results
rmse_temp <- rep(0,k_inner)
best_parameter_temp <- matrix(rep(0,k_inner*numOfParameter),k_inner,numOfParameter)
colnames(best_parameter_temp) <- c("mtry","ntree","max_nodes")
# draw random integers for the k-folds
folds_inner <- sample(rep(1:k_inner, length.out = nrow(train)))
# start crossvalidation loop
for(k_1 in 1:k_inner){
  # split into training and validation set
  indexValidation <- which(folds_outer == k_1)
  training <- train[-indexValidation,]
  validation <- train[indexValidation,]
  
  # start GridSearch loop 
  for(i in 1:nrow(rfGrid)){
    # fit the randomForrest
    rfFit <- randomForest(y~ .,  
                          data=training[,-1], 
                          mtry = rfGrid$mtry[i],  
                          ntree = rfGrid$ntree[i] 
                          )
    # predict SalePrice
    yhat <- predict(rfFit, newdata = validation[,-1])
    # fill the first column of this matrix with the rmse results (of the log outputs)
    tuning_results[i,1] <- rmse_log(validation$y,yhat)
  }#end GridSearch
  # find the index of the best model 
  idx_best <- which(tuning_results$rmse == min(tuning_results$rmse))
  # get rmse and the parameter of the best model
  best_results <- tuning_results[idx_best,]
  # save best results
  rmse_temp[k_1] <- best_results$rmse # best rmse
  best_parameter_temp[k_1,] <- cbind(best_results$mtry,
                                     best_results$ntree,
                                     best_results$max_nodes
                                     ) 
  
  # print temporary results
  cat("\n")
  cat("RMSE:",   best_results$rmse,
      "mtry:",   best_results$mtry,
      "ntree:",  best_results$ntree,
      "max_nodes:", best_results$max_nodes,
      "inner fold:", k_1, "out of", k_inner
  )
}#end inner cv




