############### principal component to naive_preprocessing ###########

# function to perform pca
pca <- function(x, contained_variance = 0.85){
  pca <- princomp(x, cor = TRUE)
  idx <- which(cumsum(pca$sdev)/sum(pca$sdev) > contained_variance)
  x_pca <- data.frame(pca$scores)
  return(x_pca[ , -c(idx)])
}


