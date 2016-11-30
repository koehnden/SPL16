### Script that includes all the library that need to be installed
# Please add packages here if you use a new one

libraries = c('caret','elasticnet','Hmisc',
              'dummies','usdm','randomForest',
              'foreach','xgboost','Matrix',
              'ggplot2','VIM','plotly',
              'corrgram')
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
    install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

#library(caret) # general machine learning framework
#library(elasticnet) # for running ridge regression
#library(Hmisc) # for imputation function
#library(dummies) # to convert categoricals into dummies
#library(usdm) # to calculate the stepwise vif score and exclude dummies accordingly
#library(randomForest) # to implement random forrest
#library(foreach) # to paraellize for loops 
#library(xgboost) # stoachstic gradient boost
#library(Matrix) # to implement xgboost
#library(ggplot2) # to plot stuff
#library(VIM)
#library(plotly)

