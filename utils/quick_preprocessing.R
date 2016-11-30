############################ Function to applies quick preprocessing ######################################
# load data set
source("load_ames_data.R")
# apply quick preprocessing done in quick_imputation, cat_to_dummies and delete_nz_variables in one step
# input: X_com - complete feature matrix with feature in train and test set (return by load_ames_data.R)
#        y - vector of labels in the training set
#        scaler - determines whether z_score (gaussian) or min max should be used for scaling
# output: a preprocessed version of the training set including the labels
naive_preprocessing <- function(X_com,y, scaler = "gaussian"){
  source("Data_Cleaning/convert_categoricals.R")
  source("Data_Cleaning/impute_data.R")
  source("outlier/impute_outliers.R")
  source("Data_Cleaning/scale_data.R")
  source("Feature_Selection/delete_nearzero_variables.R") # Put ouT X_com as the cleaned Feature Matrix
  X_imputed <- naive_imputation(X_com)
  X_no_outlier <- data.frame(lapply(X_imputed, iqr_outlier))
  X_scaled <- scale_data(X_no_outlier, scale_method = scaler)
  X_encoded <- data.frame(lapply(X_scaled, cat_to_dummy))
  X_com <- delect_nz_variable(X_encoded)
  # remerge train data 
  train <- cbind(X_com[1:length(y),],y)
  return(list(train=train[,-1],X_com=X_com)) # return without id 
}

# naive_preprocessing + converted rating scores (see above)
basic_preprocessing <- function(X_com,y, scaler = "gaussian"){
  source("Data_Cleaning/replace_ratings.R")
  source("Data_Cleaning/convert_categoricals.R")
  source("Data_Cleaning/impute_data.R")
  source("Data_Cleaning/encode_time_variables.R")
  source("outlier/impute_outliers.R")
  source("Data_Cleaning/scale_data.R")
  source("Feature_Selection/delete_nearzero_variables.R") # Put ouT X_com as the cleaned Feature Matrix
  X_com <- replace_ratings(X_com)
  X_imputed <- naive_imputation(X_com)
  X_time_encoded <- include_quarter_dummies(X_imputed)
  X_no_outlier <- data.frame(lapply(X_time_encoded, iqr_outlier))
  X_scaled <- scale_data(X_no_outlier, scale_method = scaler)
  X_encoded <- data.frame(lapply(X_scaled, cat_to_dummy))
  X_com <- delect_nz_variable(X_encoded)
  # remerge train data
  idx_train <- c(1:length(y))
  train <- cbind(X_com[idx_train,],y)
  test <- X_com[-idx_train,]
  return(list(train=train,X_com=X_com, test=test)) # return without id 
}


## example
#train <- naive_preprocessing(X_com,y)

# applies for sophisticated preprocessing done in cat_to_dummies and delete_nz_variables in one step
# input: X_imputed - complete feature matrix with feature in train and test set where NAs are already imputed by some method
#        y - vector of labels in the training set
# output: a preprocessed version of the training set including the labels
preprocessing <- function(X_imputed,y){
  source("Data_Cleaning/convert_categoricals.R")
  source("Feature_Selection/delete_nearzero_variables.R") # Put ouT X_com as the cleaned Feature Matrix
  X_encoded <- data.frame(lapply(X_imputed, cat_to_dummy))
  X_no_outlier <- data.frame(sapply(X_encoded, iqr_outlier))
  X_scaled <- scale_data(X_no_outlier, scale_method = "gaussian")
  X_com <- delect_nz_variable(X_scaled)
  
  # remerge train data 
  train <- cbind(X_com[1:length(y),],y)
  return(train)
}

## example
#X_imputed <- read.csv("Data/ames_imputed.csv")
#train <- preprocessing(X_imputed,y)
