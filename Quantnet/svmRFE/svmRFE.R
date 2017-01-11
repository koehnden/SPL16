########################### SMV feature selection #########################

### setwd('F:/PHD/IRTG/courses/SPL/Quantlet/svmRFE')
rm(list = ls())
graphics.off()

libraries = c("caret", "ggplot2")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
    install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

load("basic_processing.RData")

train = basic_data$train
y = train$y
train$y = NULL

# set cv parameter
k = 5  # folds on the cv loop

subsets = 30:99
svmGaussianGrid = expand.grid(C = 4.5, sigma = 0.002)
ctrl = rfeControl(functions = caretFuncs, method = "cv", number = k, returnResamp = "final", verbose = TRUE)

rfe_gaussiabSVM = rfe(x = train, y = y, sizes = subsets, rfeControl = ctrl, method = "svmRadial", 
    metric = "RMSE", tuneGrid = svmGaussianGrid)

feature_ranking = predictors(rfe_gaussiabSVM)
ggplot(rfe_gaussiabSVM, type = c("g", "o"))
