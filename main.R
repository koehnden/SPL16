#preparation
setwd("/Users/d065820/Documents/HU_Belrin/SPL/SPL16/Data")

#load data
train <- read.csv("ames_train.csv", header=T)
test <- read.csv("ames_test.csv", header=T)

# split target variable and feature matrix
y <- train[,81] # target variable SalePrice
X <- train[,-81] # feature matrix without target variable
# merge test and train features to get the complete feature matrix
X_com <- rbind(X,test)
##########


#####converting vars######
source("../Data_Cleaning/convert_categoricals.R")
source("../Data_Cleaning/impute_data.R")

X_imputed <- quick_imputation(X_com)
X_encoded <- data.frame(lapply(X_imputed, cat_to_dummy))






