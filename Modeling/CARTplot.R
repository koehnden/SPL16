############################# plot a decision tree ##########################################
library(rpart)				        # Popular decision tree algorithm
library(caret)	
library("rpart.plot")
library(Rmisc)

source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")
# get preprocessed data
preprocessed_data <- tree_preprocessing(X_com,y)
# train set
train <- preprocessed_data$train
# test set
test <- preprocessed_data$test

folds <- sample(rep(1:2, length.out = nrow(train)))
index1 <- which(folds ==1)
index2 <- which(folds ==2)

# run cart on two different subsets
dtree1 <- rpart(y ~ ., data=train[index1,]) # create decision tree classifier
dtree2 <- rpart(y ~ ., data=train[index2,]) # create decision tree classifier

# plot decision tree
par(mfrow=c(1, 2))
prp(dtree1, fallen.leaves = FALSE, type=0, extra=0, varlen=0, faclen=0, cex = 1.1)
prp(dtree2, fallen.leaves = FALSE, type=0, extra=0, varlen=0, faclen=0, cex = 1.1)