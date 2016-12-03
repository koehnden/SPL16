########################### Variable Importance and Feature Selection using xgboost #############################
library(caret) 
library(xgboost)
library(Matrix)
library(Ckmeans.1d.dp)
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")  # to get performance metrics 
# get preprocessed data
train <- basic_preprocessing(X_com,y)$train
# set labels and exclude them from the training set
y <- train$y
train$y <- NULL
# convert for data into a format xgb.train can handle
dtrain <- xgb.DMatrix(data = sapply(train, as.numeric), label=y)

# determine arbitrary xgboost parameters in a list
xgb_paramters = list(                                              
    eta = 0.025,                                  # learning rate                                                                
    max.depth = 12,                  # max nodes of a tree                                                       
    gamma = 2,                          # minimal improvement per iteration
    colsample_bytree = 0.8,    # fraction of variable to consider per tree (similar to mtry in rf)
    subsample = 0.4,                  # fraction of the whole sample that the bootstrap sample should consist of 
    eval_metric = "rmse",                                       # error metric
    maximize = FALSE
)
# fit the xgboost
xgbFit <- xgb.train(params = xgb_paramters,  # list of parameter previously specified
                    data =  dtrain,
                    booster = "gbtree",
                    nround = 500,    #number of trees (set accordingly to tune_nrounds function) 
                    verbose = 1,
                    objective = "reg:linear"
)

#train$data@Dimnames[[2]] represents the column names of the sparse matrix.
importance_matrix <- xgb.importance(colnames(train), model = xgbFit)
xgb.plot.importance(importance_matrix)
write.csv(importance_matrix, "Features_Selection/variable_importance_basic.csv")


retained_variables <- 1:nrow(importance_matrix)
variance_level <- cumsum(importance_matrix$Gain)
retained <- data.frame(variance_level,retained_variables)
p <- ggplot(data = retained, mapping = aes(retained_variables,variance_level)) 
p <- p + geom_line() + geom_point()
p <- p + ggtitle("Retained Variables vs. Cumsum VI") + xlab("# Variables") + ylab("Cumsum of VI")
print(p)

