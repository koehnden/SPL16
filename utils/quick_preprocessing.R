############################ Function to applies quick preprocessing ######################################
# load data set
source("load_ames_data.R")
# apply quick preprocessing done in quick_imputation, cat_to_dummies and delete_nz_variables in one step
# input: X_com - complete feature matrix with feature in train and test set (return by load_ames_data.R)
#        y - vector of labels in the training set
# output: a preprocessed version of the training set including the labels
naive_preprocessing <- function(X_com,y){
  source("Data_Cleaning/convert_categoricals.R")
  source("Data_Cleaning/impute_data.R")
  source("Feature_Selection/delete_nearzero_variables.R") # Put ouT X_com as the cleaned Feature Matrix
  X_imputed <- naive_imputation(X_com)
  X_encoded <- data.frame(lapply(X_imputed, cat_to_dummy))
  X_com <- delect_nz_variable(X_encoded)
  
  # remerge train data 
  train <- cbind(X_com[1:length(y),],y)
  return(train)
}

## example
train <- naive_preprocessing(X_com,y)

# applies for sophisticated preprocessing done in cat_to_dummies and delete_nz_variables in one step
# input: X_imputed - complete feature matrix with feature in train and test set where NAs are already imputed by some method
#        y - vector of labels in the training set
# output: a preprocessed version of the training set including the labels
preprocessing <- function(X_imputed,y){
  source("Data_Cleaning/convert_categoricals.R")
  source("Feature_Selection/delete_nearzero_variables.R") # Put ouT X_com as the cleaned Feature Matrix
  X_encoded <- data.frame(lapply(X_imputed, cat_to_dummy))
  X_com <- delect_nz_variable(X_encoded)
  
  # remerge train data 
  train <- cbind(X_com[1:length(y),],y)
  return(train)
}

## example
#X_imputed <- read.csv("Data/ames_imputed.csv")
#train <- preprocessing(X_imputed,y)
