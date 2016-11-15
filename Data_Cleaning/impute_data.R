################### Quick Data Cleaning ####################################
# functions only work if Hmisc package is installed
library(Hmisc) # for imputation function
library(mice)
library(VIM)
#library(xtabel)

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
median_mode_impute <- function(x) {
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
naive_imputation <- function(X){
  temp <- X
  # Four Variables namely "Allez", "PoolQC" ,"Fence", "MiscFeature" have over 2000 missings --> we drop them
  temp <- temp[ , !names(temp) %in% c("Alley", "PoolQC" ,"Fence", "MiscFeature")] 
  # apply function on all variable using lappy 
  temp <- data.frame(temp, stringsAsFactors = FALSE)
  temp <- as.data.frame(lapply(temp, median_mode_impute)) 
  return(temp)
}
# usage
X_imputed <- naive_imputation(X_com)

# function to impute a whole feature matrix using the mice package defaults: numerical with pmm
# and categoricals with log/poly reg, This take a while. The imputed data set can be found 
# input: X - a data.frame  
# output: data.frame X with imputed NAs by either the mean or the mode
mice_imputation <- function(x){
  temp <- mice(x, MaxNWts=5000) # MaxNWt from the nnet package
  temp <- complete(temp,1)
  return(temp)
}

#result_path <- "Data/ames_imputed_mice.csv"
#X_imputed_mice <- mice_imputation(X_com)
#write.csv(X_imputed_mice,result_path)

## usage
# apply quick_impute function to the complete feature matrix X_com
# run load_ames_data script
#source("../load_ames_data.R")
#X_imputed <- quick_imputation(X_com)
## check if function works
#sapply(X_imputed, function(x) sum(is.na(x)))
#str(X_imputed)


