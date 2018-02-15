# =================================================================================================================
# Random Forest
# =================================================================================================================

library(caret)
library(randomForest)
library(dplyr)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

# removing row with NA for country destination 
train <- train[-which(is.na(train$country_destination)), ]
# removing first column 
train <- train[,-1]
# converting all column that are integers to numeric (for xgb)
train <- mutate_if(train, is.integer, as.numeric)

# creating train/test sets
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
training <- train[train_index, ]
testing <- train[-train_index, ]

rf_model <- train(country_destination ~ ., data = training, method = "rf", prox = TRUE)
