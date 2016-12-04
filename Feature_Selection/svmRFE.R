###########################Load Data Cleaning Scripts
source("load_ames_data.R")
source("utils/quick_preprocessing.R") # to perform the naive preprocessing step implemented in the beginning
source("utils/performanceMetrics.R")  # to get performance metrics 
# get preprocessed data
train <- basic_preprocessing(X_com,y)$train
y <- train$y
train$y <- NULL

# set cv parameter
k <- 5 # folds on the cv loop

library(caret)
subsets <- 30:99
svmGaussianGrid <- expand.grid(C = 4.5, sigma = 0.002) 
ctrl <- rfeControl(functions=caretFuncs, 
                   method = "cv",
                   number = k, 
                   returnResamp="final", 
                   verbose = TRUE
                   )

rfe_gaussiabSVM <- rfe(x = train,
                       y = y,
                       sizes=subsets,
                       rfeControl=ctrl,
                       method="svmRadial",
                       metric = "RMSE",
                       tuneGrid = svmGaussianGrid # grid
)

predictors(rfe_gaussiabSVM)
ggplot(rfe_gaussiabSVM, type = c("g", "o"))
