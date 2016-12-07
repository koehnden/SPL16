############################ Tuning Random Forrest using H20 ###################################
library(h2o)
source("load_ames_data.R")
source("utils/quick_preprocessing.R")
source("utils/performanceMetrics.R")  # to get performance metrics

result_path <- "Modeling/Results/rf/rf_pca"

## Create an H2O cloud 
h2o.init(
  nthreads=-1,            ## -1: use all available threads (use all cores)
  max_mem_size = "2G")    ## specify the memory size for the H2O cloud
h2o.removeAll() # Clean slate - just in case the cluster was already running

# get preprocessed training set
train <- pca_preprocessing(X_com,y,0.8)$train 
test <- pca_preprocessing(X_com,y,0.8)$test

# specify repetions and number of folds
repetitions <- 2
k_folds <- 5

# convert into h20_objects
train_h2o <- as.h2o(train)
# sepecify columns of inputs and labels
col_label <- which(colnames(train) == "y")
col_input <- which(colnames(train) != "y")


# specify grid
rfGrid <- list(mtries = seq(10,30,5),
               ntrees = 500,
               max_depth = seq(5,25,5),
               sample_rate = seq(0.6,0.8,0.2),
               stopping_metric = "MSE",
               stopping_rounds = 20
)
# draw new seeds in order to run a different cv split each repetition
seeds <- sample(1:1000,repetitions)

for(t in 1:repetitions){
  #tune random forest with h2o
  rfFit <- h2o.grid("randomForest",
                    grid_id = "gridSearch",
                    x = col_input, 
                    y = col_label, 
                    training_frame = train_h2o,
                    hyper_params = rfGrid,
                    nfolds = k_folds,
                    is_supervised = TRUE,
                    seed = seeds[1]
  )
  # get tuning results and save them as csv
  tuning_results <- h2o.getGrid(grid_id = "gridSearch", sort_by = "rmse")
  write.csv(tuning_results@summary_table,file = paste(result_path,".csv", sep="_"))
}

model_ids <- rfFit@model_ids
models <- lapply(model_ids, function(id) { h2o.getModel(id)})
