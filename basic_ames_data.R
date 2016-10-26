##################### load ames housing data #######################

# set working directory and load data
setwd("/Users/linxi/Documents/SPL project/Data")
train =read.csv("ames_train.csv", header=T)
test = read.csv("ames_test.csv", header=T)

# split target variable and feature matrix
y = train[,81] # target variable SalePrice
X = train[,-81] # feature matrix without target variable

# merge test and train features to get the complete feature matrix
X_com = rbind(X,test)


# see the number of NA of each variables
colSums(sapply(train, is.na))
# see the number of house that were remodeled
sum(train[,'YearRemodAdd'] != train[,'YearBuilt'])

# see the percentage of house remodeled
sum(train[,'YearRemodAdd'] != train[,'YearBuilt'])/dim(train)[1]

# extract all the character variables and numerical variables
char_var = names(train)[which(sapply(train, is.character))]
numeric_var = names(train)[which(sapply(train, is.numeric))]

# summary of character variables and numerical variables
summary(train[char_var])
summary(train[numeric_var])

# summary of train data
summary(train)

# insight of train and test data (how many rows and colums)
dim(train)[1]
dim(train)[2]
dim(test)[1]
dim(test)[2]
