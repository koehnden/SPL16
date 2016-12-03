############################ Tuning Generalized Linear Models using H20 #####################################################
library(h2o)
source("load_ames_data.R")
source("utils/quick_preprocessing.R")
source("utils/performanceMetrics.R")  # to get performance metrics

result_path <- "Modeling/Results/glm/basic_preprocessing"

## Create an H2O cloud 
h2o.init(
  nthreads=-1,            ## -1: use all available threads (use all cores)
  max_mem_size = "2G")    ## specify the memory size for the H2O cloud
h2o.removeAll() # Clean slate - just in case the cluster was already running

# get preprocessed training set
train <- basic_preprocessing(X_com,y)$train

# specify repetions and number of folds
repetitions <- 5
k_folds <- 5

# convert into h20_objects
train_h2o <- as.h2o(train)
# sepecify columns of inputs and labels
col_label <- which(colnames(train) == "y")
col_input <- which(colnames(train) != "y")


# specify grid for LASSO (alpha = 1)
lassoGrid <- list(lambda = seq(50,300,5),  # L1-Norm Penalty
               alpha = 1                  # LASSO
)

ridgeGrid <- list(lambda = seq(0.001,0.1,0.001),  # L2-Norm Penalty
                 alpha = 0                        # Ridge
)
length(lassoGrid$lambda)

# draw new seeds in order to run a different cv split each repetition
seeds <- sample(1:1000,repetitions)

model <- "lasso"
for(t in 1:repetitions){
  #tune random forest with h2o
    glmFit <- h2o.grid("glm",
                    grid_id = model,
                    x = col_input, 
                    y = col_label, 
                    training_frame = train_h2o,
                    hyper_params = lassoGrid,
                    nfolds = k_folds,
                    is_supervised = TRUE,
                    seed = seeds[t]
  )
    
  # get tuning results and save them as csv
  tuning_results <- h2o.getGrid(grid_id = model, sort_by = "rmse")
  write.csv(tuning_results@summary_table,file = paste(result_path,model,".csv", sep="_"))
}