setwd('~/Documents/pima_indian/')
library(matrixStats)
library(caTools)
library(caret)
set.seed(42)

# Importing the dataset
dataset <- read.csv('pima-indians-diabetes.csv')

# Set up how many times that we are going to run the code
trials <- 10

acc_list <- rep(0, trials)

# Getting some essential constant here
num.features <- ncol(dataset)-1
# View the y label name
label.name <- colnames(dataset)[ncol(dataset)]

for (each in 1: trials){
  
  trainIndex <- createDataPartition(dataset$V9, p = 0.8, list = FALSE)
  train <- dataset[ trainIndex,]
  test <- dataset[-trainIndex,]
  numTrain <- nrow(train)
  
  # train.label0 <- train[which(train$V9 == 0),]
  # train.label1 <- train[which(train$V9 == 1),]
  train.label0 <- as.matrix(train[which(train$V9 == 0),])[, 1:num.features]
  train.label1 <- as.matrix(train[which(train$V9 == 1),])[, 1:num.features]
  
  # Getting priors
  train.label0.prior <- nrow(train.label0) / numTrain
  train.label1.prior <- nrow(train.label1) / numTrain
  
  # Assuming all features across samples are represented as normal model
  # Getting mean and standard deviation
  train.label0.expect <- colMeans(train.label0)
  train.label1.expect <- colMeans(train.label1)
  
  is.numeric(train.label0.std)
  
  train.label0.std <- colSds(train.label0)
  train.label1.std <- colSds(train.label1)
  
  # Creating new column in 
  test$predict <- NA
  
  # Making prediction
  for (idx in 1:nrow(test)){
    
    x <- as.numeric(test[idx, 1:num.features])
    pred.zero <-sum(dnorm(x, mean = train.label0.expect, sd = train.label0.std, log = TRUE)) + log(train.label0.prior)
    pred.one <- sum(dnorm(x, mean = train.label1.expect, sd = train.label1.std, log = TRUE)) + log(train.label1.prior)
    if(pred.one >= pred.zero){
      test[idx, ]$predict <- 1
    } else {
      test[idx, ]$predict <- 0
    }
  }
  
  acc_list[each] <- sum(test$V9 == test$predict) / nrow(test)
}

