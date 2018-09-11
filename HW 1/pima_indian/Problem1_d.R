setwd('~/Documents/pima_indian/')
path = '/home/ting/svm_light/'
dataset <- read.csv('pima-indians-diabetes.csv', header = FALSE)
library(klaR)
library(caret)
set.seed(42)

# Set up how many times that we are going to run the code
trials <- 10

acc_list <- rep(0, trials)

# Getting some essential constant here
num.features <- ncol(dataset)-1

for (each in 1: trials){

  trainIndex <- createDataPartition(dataset$V9, p = 0.8, list = FALSE)
  train <- dataset[ trainIndex,]
  test <- dataset[-trainIndex,]

  train.feature <- train[, -c(9)]
  train.label <- train[, 9]
  test.feature <- test[, -c(9)]
  test.label <- test[, 9]
  num_test <- nrow(test)

  svm_model <- svmlight(train.feature, train.label, pathsvm = path)

  prediction <- predict(svm_model, test.feature)

  acc_list[each] <- sum(test.label == prediction$class) / num_test
}

cat('The average accuracy over 10 times: ', mean(acc_list))

# Result: The average accuracy over 10 times:  0.7300654
