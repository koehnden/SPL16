############### principal component to naive_preprocessing ###########
source("utils/quick_preprocessing.R")

data_basic = basic_preprocessing(X_com, y)
X_com_basic = data_basic$X_com

pca <- function(x, contained_variance = 0.85){
  pca <- princomp(x, cor = TRUE)
  idx <- which(cumsum(pca$sdev)/sum(pca$sdev) > contained_variance)
  x_pca <- data.frame(pca$scores)
  return(x_pca[ , -c(idx)])
}

