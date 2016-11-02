############################### Random Forrest  #############################################################
# This script tunes the parameter of a Random Forrest using GridSearch and cross-validation 
# Depending on the parameter set this takes quite a while
library(randomForest)
library(foreach) # to paraellize for loops (to be implemented)
library(doParallel) # to register the cores as a cluster 

number_of_cores <- 4 # change the 4 to your number of CPU cores
cl<-makeCluster(number_of_cores) 
registerDoParallel(cl) # to register DOParaellel

########## perform repeated nested cv
# set cv parameter
t_outer <- 5 # repetitions on the outer loop (not implemented)
k_outer <- 5 # fold of the outer cv loop (not yet implemented)
t_inner <- 5 # repetition on inner loop (not implemented)
k_inner <- 5 # folds on the inner cv loop

# create Grid for GridSearch to tune hyperparameter (here just lambda)
# Parameters: mtry: variable considered at each split
#             ntree: number of tree to grown 
#             maxnodes: longest path of a single tree
#             samplesize: size of the bagging bootstrap sample (not yet inspected)
rfGrid <-  expand.grid(mtry = seq(10,50,10), ntree = 500, maxnodes = seq(5,25,5))  
# samplesize could be inspected as well
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
for (k_1 in 1:k_inner) {
  # split into training and validation set
  indexValidation <- which(folds_inner == k_1)
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


stopCluster(cl) # stop the cluster  



