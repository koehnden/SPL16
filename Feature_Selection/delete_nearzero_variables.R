############################### delete near zero variance variable #######################3
library(caret)
#source("~/SPL16/Data_Cleaning/convert_categoricals.R")
# Function to delete variable with no or almost no variance 
# imput: X - a data.frame or matrix that include the features
# output a reduced feature martix without near zero variance
delect_nz_variable <- function(X){
  near_zero_variance <- nearZeroVar(X)
  return(X[,-near_zero_variance])
}
# usage
#X_com <- delect_nz_variable(X_encoded)