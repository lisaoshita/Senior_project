# =================================================================================================================
# Random Forest
# =================================================================================================================

library(caret)
library(randomForest)
library(dplyr)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

# removing first column 
train <- train[,-1]
# converting all column that are integers to numeric
train <- mutate_if(train, is.integer, as.numeric)

# setting up training and test data

# training data 
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
train_data <- train[train_index[,1], ]

# test data 
test_data <- train[-train_index[,1], -1]
test_label <- train[-train_index[,1], 1]

# random forest model 
rf_model <- randomForest(country_destination ~ ., data = train_data, ntree = 10, importance = TRUE, sampsize = 100)
rf_model

# predicted response
pred <- predict(rf_model, newdata = test_data)
table(pred, test_label)

sum(pred == test_label) / length(pred)
# 87% accuracy 
