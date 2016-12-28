################################# Ridge Regression ######################################################################
# This script performs repeated cross-validation to tune SVM-Regression. 
library(caret)
###########################Load Data Cleaning Scripts
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")  # to get performance metrics 
# get preprocessed data
train <- basic_preprocessing(X_com,y)$train
y <- train$y
train$y <- NULL

# set cv parameter
t <- 10 # repetition on inner loop (here caret does it)
k <- 5 # folds on the inner cv loop

# create Grid for GridSearch to tune hyperparameter 
svmLinearGrid <-  expand.grid(C = c(0.001,0.01,0.1,1)) 
svmGaussianGrid <- expand.grid(C = c(0.001,0.01,0.1,1), sigma = c(0.001,0.01,0.1,1)) 

# determine evaluation method of the inner cv loop
ctrl <- trainControl(method="repeatedcv",
                     number=t,
                     repeats=k,
                     verboseIter=FALSE
)

# SVM with linear kernel
svmLinearFit <- train(x = train,
                  y = y,
                  method = 'svmLinear',  # method
                  trControl = ctrl,  # evaluatio method (repeated CV)
                  tuneGrid = svmLinearGrid, # grid
                  metric = "RMSE",  # error metric
                  maximize = FALSE
)
plot(svmLinearFit)

# SVM with Gaussian Kernel
svmGaussianFit <- train(x = train,
                      y = y,
                      method = 'svmRadial',  # method
                      trControl = ctrl,  # evaluatio method (repeated CV)
                      tuneGrid = svmGaussianGrid, # grid
                      metric = "RMSE",  # error metric
                      maximize = FALSE
)
plot(svmGaussianFit)