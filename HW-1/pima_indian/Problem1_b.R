# Pretty much the same as the first one except that value 0 need to be treated as NA
# attribute 3 (Diastolic blood pressure), attribute 4 (Triceps skin fold thickness), 
# attribute 6 (Body mass index), and attribute 8 (Age)

setwd('~/Documents/pima_indian/')
library(matrixStats)
library(caTools)
library(caret)
set.seed(42)

# Importing the dataset
dataset <- read.csv('pima-indians-diabetes.csv', header = FALSE)


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
  
  # Filling 0 with NA
  train$V3[which(train$V3 == 0)] <- NA
  train$V4[which(train$V4 == 0)] <- NA
  train$V6[which(train$V6 == 0)] <- NA
  train$V8[which(train$V8 == 0)] <- NA
  
  
  train.label0 <- as.matrix(train[which(train$V9 == 0),])[, 1:num.features]
  train.label1 <- as.matrix(train[which(train$V9 == 1),])[, 1:num.features]
  
  # Getting priors
  train.label0.prior <- nrow(train.label0) / numTrain
  train.label1.prior <- nrow(train.label1) / numTrain
  
  # Assuming all features across samples are represented as normal model
  # Getting mean and standard deviation
  train.label0.expect <- colMeans(train.label0, na.rm = TRUE)
  train.label1.expect <- colMeans(train.label1, na.rm = TRUE)
  
  train.label0.std <- colSds(train.label0, na.rm = TRUE)
  train.label1.std <- colSds(train.label1, na.rm = TRUE)
  
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

cat('The average accuracy over 10 times: ', mean(acc_list))

# Result: The average accuracy over 10 times:  0.7098039