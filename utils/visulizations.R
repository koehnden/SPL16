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
#price_per_factor_box(train$MSZoning, "MSSubClass")

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