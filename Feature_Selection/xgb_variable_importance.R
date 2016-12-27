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
    max.depth = 16,                  # max nodes of a tree                                                       
    gamma = 0,                          # minimal improvement per iteration
    colsample_bytree = 0.8,    # fraction of variable to consider per tree (similar to mtry in rf)
    subsample = 0.6,                  # fraction of the whole sample that the bootstrap sample should consist of 
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
write.csv(importance_matrix, "Features_Selection/variable_importance_basic.csv")

### customized vimp_plot
library(ggplot2)
ranked_variables <- importance_matrix$Feature
importance_per <- importance_matrix$Gain
var_imp <- data.frame(ranked_variables,importance_per)[1:30,]
p <- ggplot(var_imp, aes(x=reorder(ranked_variables, importance_per), weight=importance_per, fill=ranked_variables))
p <- p + geom_bar(aes(weights=importance_per)) + xlab("Importance Score") + ylab("Variables") +
  ggtitle("Variable Importance from Gradient Boosting Fit") + theme(legend.position="none") + coord_flip()
print(p)

# plot retained variable vs. cumsum variable importance
retained_variables <- 1:nrow(importance_matrix)
variance_level <- cumsum(importance_matrix$Gain)
retained <- data.frame(variance_level,retained_variables)
p <- ggplot(data = retained, mapping = aes(retained_variables,variance_level)) 
p <- p + geom_line() + geom_point()
p <- p + ggtitle("Retained Variables vs. Cumsum VI") + xlab("# Variables") + ylab("Cumsum of VI")
print(p)
