##################### load ames housing data #######################

# set working directory and load data
setwd("/Users/linxi/Documents/SPL16/Data")
train =read.csv("ames_train.csv", header=T,stringsAsFactors = F)
test = read.csv("ames_test.csv", header=T,stringsAsFactors = F)

# split target variable and feature matrix
y = train[,81] # target variable SalePrice
X = train[,-81] # feature matrix without target variable

# merge test and train features to get the complete feature matrix
X_com = rbind(X,test)

############## seperate the numeric variables and character variables ###################2
 # see the number of NA of each variables
colSums(sapply(X_com, is.na))

 # extract all the character variables and numerical variables
char_var_list = names(X_com)[sapply(X_com, is.character)]
numeric_var_list = names(X_com)[sapply(X_com, is.numeric)]

 # seperate into two tables
char_var = X_com[,char_var_list] 
numeric_var = X_com[,numeric_var_list]
 
 # find outlier in numeric variables
summary(numeric_var)
table(colSums(sapply(numeric_var, is.na)))


######################### impout outlier with upper and lower bounds #####################3
impute_outlier = function(x) {
  #trial = X_com$MSSubClass
  bp = boxplot(x)
  lower = bp$stats[1]; upper = bp$stats[5]
  x[x>upper] = upper
  x[x<lower] = lower
  return(x)
}

 # new data frame 
noOut_num = data.frame(lapply(numeric_var,impute_outlier))
#table(colSums(sapply(noOut_num, is.na)))


############################### delete near zero variance variable #######################4
library(caret)

delect_nz_variable <- function(X){
  near_zero_variance <- nearZeroVar(X)
  return(X[,-near_zero_variance])
}

noOut_num = delect_nz_variable(noOut_num)



############################### insert values to NA ####################################5
library(mice)
library(VIM)
library(xtabel)

NA_pattern = t(md.pattern(noOut_num))
#NA_pattern = xtable(t(md.pattern(noOut_num)))
#print(NA_pattern, file = "/Users/linxi/Documents/SPL16/outlier/result_of_NA_pattern")

# visual representation for missing data pattern
dev.new()
aggr_plot = aggr(noOut_num, col=c('navyblue','red'), numbers=TRUE, 
                  sortVars=TRUE, labels=names(data), cex.axis=.7, 
                  gap=3, ylab=c("Histogram of missing data","Pattern"))

####################### impute missing values based on mice #############################6
NAimpute_noOut_num = mice(noOut_num, m=5, method = 'pmm', seed = 500)
completed_NAimpute = complete(NAimpute_noOut_num,1)
table(colSums(sapply(completed_NAimpute, is.na)))


########## inspecting the distribution of original and imputed data ######################7
dev.new()
densityplot(NAimpute_noOut_num)

dev.new()
stripplot(NAimpute_noOut_num, pch = 20, cex = 1.2)




