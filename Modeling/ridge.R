######################## Ridge Regression ########################################
library(caret)
#library(doMC)
#registerDoMC(cores = 4)  # only available in linux

# run Data Cleaning Scripts
setwd("~/SPL16")
source("~/SPL16/Data_Cleaning/convert_categoricals.R")
source("~/SPL16/Feature_Selection/delete_nearzero_variables.R") # Put ouT X_com as the cleaned Feature Matrix

# remerge train data 
train <- cbind(X_com[1:length(y),],y)
# split into train and validation (simple method) --> TODO: run another CV loop here to do nested CV
set.seed(100)
# strati???ed random split of the data
indexTrain <- createDataPartition(y = train$y, p = .8,list = FALSE)
training <- train[indexTrain,]
validation <- train[-indexTrain,]

# create Grid for GridSearch Parameter optimizing
ridgeGrid <-  expand.grid(lambda = seq(0.02,0.03,0.0005)) # Grid already specialized

ctrl <- trainControl(method = "repeatedcv",
                     number = 10, # how many folds (k)
                     repeats = 5  # how many repitions (t)
                    #allowParallel=TRUE 
                    )

# takes a couple of minutes on my machine (32RAM) without paraellization --> quite expensive
ridgeFit <- train(y ~., 
                  data = training[,-1], # exclude Id Variable from training data
                  method = 'ridge',  # method
                  trControl = ctrl,  # evaluatio method (repeated CV)
                  tuneGrid = ridgeGrid, # grid
                  metric = "RMSE"  # error metric
                  # verbose = True # print steps
                  )

# plot learning curve
plot(ridgeFit)
## predict remaining validation labels
yhat <- predict(ridgeFit, newdata = validation)
# calculate rmse of the log data
source("~/SPL16/utils/performanceMetrics.R")
rmse_log(validation$y,yhat) # 0.08770993 way too score!!
