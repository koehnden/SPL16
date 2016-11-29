################################## Replace Time Variables #####################################################3
# This script provides function to plot a variable against the house price
library(data.table)
library(ggplot2)

price_per_factor_box <- function(factor, factor_name){
  sold_per_x <- data.frame(factor,train$SalePrice)
  colnames(sold_per_x) <- c(factor_name,"SalePrice")
  # create boxplot
  p <- ggplot(sold_per_x) + geom_boxplot(aes(x=factor, y=SalePrice, group=factor, fill=factor))
  # add title
  p <- p + ggtitle(paste(factor_name,"SalePrice", sep=" vs. ")) 
  # add colours
  p <- p + theme(legend.position="none") + xlab(factor_name) 
  return(p)
}
# usage example: boxplot of SalePrice by MSZoning 
#price_per_factor_box(train$MSZoning, "MSZoning")

price_per_factor_plot <- function(factor, factor_name){
  sold_per_x <- data.frame(factor,train$SalePrice)
  colnames(sold_per_x) <- c(factor_name,"SalePrice")
  # create scatterplot
  p <- ggplot(sold_per_x) + geom_point(aes(x=factor, y=SalePrice))
  # add mean and confidence intervall
  p <- p + geom_smooth(aes(factor))
  # add title
  p <- p + ggtitle(paste(factor_name,"SalePrice", sep=" vs. ")) + xlab(factor_name)
  return(p)
}
# usage examples: plot YrBuild vs. SalePrice
#price_per_factor_plot(train$YearBuilt,"YrBuild") 

# function to do a boxplot for a parameter
box_hyperparameter <- function(results, parameter, parameter_name){
  grid <- sort(unique(parameter))
  # create boxplot
  p <- ggplot(results) + geom_boxplot(aes(x=parameter, y=rmse, group=parameter, fill=parameter))
  # add title
  p <- p + ggtitle(paste(parameter_name,"RMSE of log y", sep=" vs. ")) 
  # add colours
  p <- p + theme(legend.position="none") + xlab(parameter_name) + scale_x_discrete(limits= c(grid))
  return(p)
}

# usage
#max_depth <- box_hyperparameter(results_inner, results_inner$max_depth,"Max Depth") 

# function to plot two different parameters in a headmap according to their rmse results
hyperparameter_heatmap <- function(results, parameter1, parameter2, name1, name2){
  grid1 <- sort(unique(parameter1))
  grid2 <- sort(unique(parameter2))
  p <- ggplot(results, aes(parameter1, parameter2)) + geom_raster(aes(fill = rmse), interpolate = F)
  p <- p + xlab(name1) + ylab(name2) 
  p <- p + scale_x_discrete(limits= c(grid1)) + scale_y_discrete(limits = c(grid2))
  return(p)
}
# usage
#p1 <- hyperparameter_heatmap(results_inner, results_inner$max_depth, results_inner$gamma, "Max Depth", "Gamma") 

# TODO:correlation plot of all variables
#library(corrplot)
#correlations <- cor(na.omit(train_cont[,-1, with = FALSE]))

# correlations
#row_indic <- apply(correlations, 1, function(x) sum(x > 0.3 | x < -0.3) > 1)

#correlations<- correlations[row_indic ,row_indic ]
#corrplot(correlations, method="square")