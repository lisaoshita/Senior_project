# =================================================================================================================
# Random Forest
# =================================================================================================================

library(caret)
library(randomForest)
library(dplyr)
library(magrittr)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

seed <- 100
set.seed(seed)

# removing first column 
train <- train[,-1]
# converting all column that are integers to numeric
train <- mutate_if(train, is.integer, as.numeric)

# =================================================================================================================

# setting up training and test data

# training data 
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
train_data <- train[train_index[,1], ]

# test data 
test_data <- train[-train_index[,1], -1]
test_label <- train[-train_index[,1], 1]

# =================================================================================================================

# random forest model 

# with default parameters
# mtry: sqrt(number of predictors)
# default ntree is 500 - takes a long time, ntree = 50 still takes a long time 

rf_model <- randomForest(country_destination ~ ., data = train_data, ntree = 50, importance = TRUE, do.trace = 10)
rf_model # error rate: 12.46%, accuracy: 87.54% (out-of-bag, similar to CV)

# predictions on test data

pred <- predict(rf_model, newdata = test_data)
table(pred, test_label) # only predicting NDF, US, other, FR

sum(pred == test_label) / length(pred)
# 87.6% accuracy 

# =================================================================================================================

# feature importance

# Mean decrease accuracy: 
# based on hypothesis that if a feature is not important, then rearranging values of that feature will not harm accuracy 
# for each tree prediction error calculated on out-of-bag portion of the data is recorded, then the same is done after permuting
# each predictor variable - difference between the two are averaged over all trees, and normalized by standard dev of differences 

# Mean decrease gini
# total decrease in node impurities from splitting on the variable, averaged over all trees

feature_imp <- rf_model$importance
feature_imp <- feature_imp[order(-feature_imp[, ncol(feature_imp)]), ] # decreasing order 

# =================================================================================================================
# new random forest (on only most important features)
# =================================================================================================================

include <- rownames(feature_imp)[feature_imp[, ncol(feature_imp)] > 5] # features to include (with meandecreasegini > 5)

new_train <- train %>% select(country_destination, include)
# save as csv file for stacking 
# write.csv(new_train, file = "rf_train.csv")

# training data 
new_train_data <- new_train[train_index[,1], ]

# test data 
new_test_data <- new_train[-train_index[,1], -1]
new_test_label <- new_train[-train_index[,1], 1]

# refit random forest model 

new_rf_model <- randomForest(country_destination ~ ., data = new_train_data, ntree = 50, importance = TRUE, do.trace = 10)
new_rf_model # 12.57% error rate, 87.43% accuracy (out-of-bag, similar to CV)

# predict on test set 
new_pred <- predict(new_rf_model, newdata = new_test_data)
table(new_pred, new_test_label)

sum(new_pred == new_test_label) / length(new_pred) # 87.5% accuracy 
# only predicting: NDF, US, FR, other, AU, ES, IT 



