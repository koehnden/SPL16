# plot the fraction of retained pc against  
library(ggplot2)

source("load_ames_data.R")
source("utils/quick_preprocessing.R")
source("PCA_pca_basic.R")
variance_level <- seq(0.5,0.95,0.01)
X_com <- basic_preprocessing(X_com,y)$X_com

retained_variables <- rep(0,length(variance_level))
for(i in 1:length(variance_level)){
  retained_variables[i] <- ncol(pca(X_com,variance_level[i]))
}

retained_pc <- data.frame(variance_level,retained_variables)
p <- ggplot(data = retained_pc, mapping = aes(retained_variables,variance_level)) 
p <- p + geom_line() + geom_point()
p <- p + ggtitle("Retained PC vs. Contained Variance") + xlab("# Principal Components") + ylab("Fraction of Variance Retained")
print(p)