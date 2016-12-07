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
  ntrees = 500,                      # number of trees 
  mtries = 20,                       # number of variable considered at each split
  sample_rate = 0.8,                 # fraction of booststrap sample
  max_depth = 25
)               
# save variable importance in a csv file
write.csv(rfFit@model$variable_importances,"Feature_Selection/rf_vi.csv")
# plot variable importance
h2o.varimp_plot(rfFit)

### customized vimp_plot
library(ggplot2)
ranked_variables <- rfFit@model$variable_importances$variable
importance_per <- rfFit@model$variable_importances$percentage
var_imp <- data.frame(ranked_variables,importance_per)[1:30,]
p <- ggplot(var_imp, aes(x=reorder(ranked_variables, importance_per), weight=importance_per, fill=ranked_variables))
p <- p + geom_bar(aes(weights=importance_per)) +
  ggtitle("Variable Importance from Random Forest Fit") + theme(legend.position="none") + coord_flip()
print(p)

retained_variables <- 1:nrow(var_imp)
variance_level <- cumsum(importance_per)
retained <- data.frame(variance_level,retained_variables)
p <- ggplot(data = retained, mapping = aes(retained_variables,variance_level)) 
p <- p + geom_line() + geom_point()
p <- p + ggtitle("Retained Variables vs. Cumsum VI") + xlab("# Variables") + ylab("Cumsum of VI")
print(p)
