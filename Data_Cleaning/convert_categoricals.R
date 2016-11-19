################# Function to encode categorical variables ########################
# Function to convert a categorical variable into dummies. It also filters out dummies that are linear dependent to other 
# according to the vif score to avoid multicolinearity
# assumes that you have installed the following libraries
library(dummies) # to convert categoricals into dummies
library(usdm) # to calculate the stepwise vif score and exclude dummies accordingly
# input:  x - a variable
#         vif_threshold - a number that determines the threshold at which vif score a dummies should 
#                         be excluded
# output : dummy variables (lineary undependent according to the vif score)
cat_to_dummy <- function(x, vif_threshold = 10){
  # check if variable is not numeric or has less than 5 levels else convert 
  if(!any(is.numeric(x)) | length(unique(x)) <= 5){
    # convert into dummy and exclude randomly drawn index
    x <- as.data.frame(dummy(x))
    # get number of column after hot encoding
    initial_colnumber <- ncol(x)
    # check if more than one dummy remains 
    if (ncol(x) > 2){
      # drop dummy that are linear dependent with other variables
      # exclude from the usdm package delets all dummies with a higher score than the vif_threshold
      x <- exclude(x, vifstep(x,th=vif_threshold))
      # if no column was delete we delete one randomnly to avoid dummy trap
      if(ncol(x) == initial_colnumber){
        random_drop <- sample(1:length(unique(x)), 1)
        x <- x[,-random_drop]
      }
    }
    if(ncol(x) == 2){
      x <- x[,-1]
    } 
  }
  return(x)
}

## usage 
#source("Data_Cleaning/impute_data.R")
#source("~/SPL16/Data_Cleaning/impute_data.R")
# apply cat_to_dummy on the whole data set
#X_encoded <- data.frame(lapply(X_imputed, cat_to_dummy))