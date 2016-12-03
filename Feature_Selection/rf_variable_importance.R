##################################### Random Forest Variable Importance ##########################################
library(h2o)
source("load_ames_data.R")
source("utils/quick_preprocessing.R")
source("utils/performanceMetrics.R")  # to get performance metrics

## Create an H2O cloud 
h2o.init(
  nthreads=-1,            ## -1: use all available threads (use all cores)
  max_mem_size = "2G")    ## specify the memory size for the H2O cloud
h2o.removeAll() # Clean slate - just in case the cluster was already running

# get preprocessed training set
train <- basic_preprocessing(X_com,y)$train

train_h2o <- as.h2o(train)
# sepecify columns of inputs and labels
col_label <- which(colnames(train) == "y")
col_input <- which(colnames(train) != "y")

# fit the Random Forest
rfFit <- h2o.randomForest(           # h2o.randomForest function
  training_frame = train_h2o,        # H2O frame for training
  x=col_input,                       # predictor columns, by column index
  y=col_label,                       # label column index
  model_id = "rf_covType_v1",        # name the model in H2O
  ntrees = 200,                      # number of trees 
  mtries = 10,                       # number of variable considered at each split
  sample_rate = 0.6                 # fraction of booststrap sample
)               
# save variable importance in a csv file
write.csv(rfFit@model$variable_importances,"Feature_Selection/rf_vi.csv")
# plot variable importance
h2o.varimp_plot(rfFit)
