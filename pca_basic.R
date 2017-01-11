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
