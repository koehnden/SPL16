############### principal component to naive_preprocessing ###########

data_naive = naive_preprocessing(X_com, y)
X_com_naive = data_naive$X_com
pca_naive = princomp(X_com_naive, cor = TRUE)
# print variance accounted for pca_naive
su_pca_naive = summary(pca_naive)
# pc loadings
loadings(pca_naive)
# pc loadings
p = plot(pca_naive, type = "lines")
# the principal components
pca_naive$scores
# plot
biplot(pca_naive)

### principle component to the data after basic_preprocessing

## principle analysis
pca_basic = princomp(X_com_bacis,cor = F)
summary(pca_basic)

## factor analysis
str(pca_basic$sdev)
View(su_pca_basic$sdev)
loadings_pca_bacis = loadings(pca_basic)
aa = unclass(loadings_pca_bacis)
sdev = pca_basic$sdev
sdper = sdev/sum(sdev)



