################### Quick Data Cleaning ####################################
# functions only work if Hmisc package is installed
library(Hmisc) # for imputation function

# function to compute the mode of a variable (used to impute categorical variables)
# input: x - a variable 
# output: mode of the variable x as a character
Mode <- function(x) {
  ux <- unique(x)
  as.character(ux[which.max(tabulate(match(x, ux)))])
}


# function to impute a variable (numerical with median and categoricals with mode)
# input: x - a variable 
# output: x with imputed NAs by either the mean or the mode
impute_variable <- function(x) {
  if(any(is.na(x))){
    if(is.numeric(x) & !is.factor(x) ){
      x <-  impute(x, fun = mean)
      return(as.numeric(x))
    } else {
      x <-  impute(x, fun = Mode)
      return(as.factor(x))
    }
  }
  return(x)
}

# function to impute a whole feature matrix (numerical with median and categoricals with mode)
# input: X - a data.frame  
# output: data.frame X with imputed NAs by either the mean or the mode
quick_imputation <- function(X){
  temp <- X
  # Four Variables namely "Allez", "PoolQC" ,"Fence", "MiscFeature" have over 2000 missings --> we drop them
  temp <- temp[ , !names(temp) %in% c("Alley", "PoolQC" ,"Fence", "MiscFeature")] 
  # apply function on all variable using lappy 
  temp <- data.frame(temp, stringsAsFactors = FALSE)
  temp <- as.data.frame(lapply(temp, impute_variable))
  return(temp)
}

## usage
# apply quick_impute function to the complete feature matrix X_com
# run load_ames_data script
#source("../load_ames_data.R")
#X_imputed <- quick_imputation(X_com)
## check if function works
#sapply(X_imputed, function(x) sum(is.na(x)))
#str(X_imputed)


