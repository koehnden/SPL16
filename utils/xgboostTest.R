
library(Matrix)
library(xgboost)
library(caret)

set.seed(3456)
trainIndex <- createDataPartition(train$Id, p = .8, list = F)
train <- train[trainIndex,]
validation <- train[-trainIndex,]

train<- as.matrix(train, rownames.force=NA)
validation <- as.matrix(validation, rownames.force=NA)

train <- as(train, "sparseMatrix")
test <- as(test, "sparseMatrix")
# Never forget to exclude objective variable in 'data option'
train_Data <- xgb.DMatrix(data = train[,2:76], label = train[,"SalePrice"])