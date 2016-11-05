##################### load ames housing data #######################

#setwd("~/SPL16/Data")
train <- read.csv("Data/ames_train.csv", header=T)
test <- read.csv("Data/ames_test.csv", header=T)

# split target variable and feature matrix
y <- train[,81] # target variable SalePrice
X <- train[,-81] # feature matrix without target variable
# merge test and train features to get the complete feature matrix
X_com <- rbind(X,test)
