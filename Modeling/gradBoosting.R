# set WD
setwd("~/SPL16")
##Oleksiy WD
setwd("/Users/d065820/Documents/HU_Belrin/SPL/SPL16/Data")

###########################Load Data Cleaning Scripts
source("../Data_Cleaning/convert_categoricals.R")
source("../Data_Cleaning/impute_data.R")
source("../Feature_Selection/delete_nearzero_variables.R") # Put ouT X_com as the cleaned Feature Matrix
source("../utils/performanceMetrics.R") # get performance Metric functions

#load data
train <- read.csv("ames_train.csv", header=T)
test <- read.csv("ames_test.csv", header=T)

# split target variable and feature matrix
y <- train[,81] # target variable SalePrice
X <- train[,-81] # feature matrix without target variable
# merge test and train features to get the complete feature matrix
X_com <- rbind(X,test)
##########

X_imputed <- quick_imputation(X_com)
X_encoded <- data.frame(lapply(X_imputed, cat_to_dummy))
X_com <- delect_nz_variable(X_encoded)

# remerge train data 
train <- cbind(X_com[1:length(y),],y)

###########################################




#XGBoost only works with numeric vectors
require(xgboost)
library("Matrix")

#prepare the data
str(train)
dim(train)
train_frame=data.matrix(train, rownames.force = NA)
train_labels=data.matrix(y, rownames.force = NA)
train <- as(train_frame, "dgCMatrix")
Y <- as(train_labels, "dgCMatrix")

test_frame=data.matrix(test, rownames.force = NA)
M_test <- as(test_frame, "dgCMatrix")


#Input Type: it takes several types of input data:
#  Dense Matrix: R's dense matrix, i.e. matrix ;
#Sparse Matrix: R's sparse matrix, i.e. Matrix::dgCMatrix ;
#Data File: local data files ;
#xgb.DMatrix: its own class (recommended).

#fit the model

# set cv parameter
t_outer <- 5 # repetitions on the outer loop
k_outer <- 10 # fold of the outer cv loop
t_inner <- 10 # repetition on inner loop (here caret does it)
k_inner <- 5 # folds on the inner cv loop

# create Grid for GridSearch to tune hyperparameter (here just lambda)
boostGrid <-  expand.grid(eta = seq(0.1, 1, 0.1) ) 

1:length(boostGrid$eta)

# set up empty matrices to be filled with rmse and best_paramer (here only lambda)
rmse_temp <- matrix(0, nrow = k_outer, ncol = t_outer)
best_eta <- matrix(0, nrow = k_outer, ncol = t_outer)
rmse_for_parameter_temp <- matrix(0, nrow = length(boostGrid$eta), ncol = length(boostGrid))



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
   y_training=Y[-indexValidation,]
   y_validation=Y[indexValidation]
   
   for (eta_index in 1:length(boostGrid$eta)){
     xgb <- xgboost(data = training,
               booster = "gbtree",
               label = y_training, 
               max.depth = 6, 
               eta = boostGrid$eta[eta_index], 
               nthread = 5,
               #number of trees 
               nround = 200,
               objective = "reg:linear")
     #predict
     yhat <- predict(xgb, validation)
     #record accuraccy for parameter value
     rmse_for_parameter_temp[eta,1] <- as.numeric(rmse_log(y_validation,yhat))
   }
   best_eta[k_1,t_1] <- as.numeric(boostGrid$eta[which.min(rmse_for_parameter_temp[,1])])
   #record the best rmse
   rmse_temp[k_1,t_1] <- rmse_for_parameter_temp[which.min(rmse_for_parameter_temp[,1])][1]
  }#end k_outer
}#end t_outer


# get average prediction error and sd
rmse_mean <- colMeans(rmse_temp, na.rm = T); mean(rmse_mean) 
rmse_sd <- apply(rmse_temp,2,sd); mean(rmse_sd)
# show best parameter choosen by the inner repeated cv 
table(best_eta)

#train boost
#objective = "binary:logistic": we will train a binary classification model ;
#max.deph = 2: the trees won’t be deep, because our case is very simple ;
#nthread = 2: the number of cpu threads we are going to use;
#nround = 2: there will be two passes on the data, the second one will enhance the model by further 


