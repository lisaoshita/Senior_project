# =================================================================================================================
# XGBoost
# =================================================================================================================

library(xgboost)
library(dplyr)
library(magrittr)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

# removing first column 
train <- train[,-1]
# converting all column that are integers to numeric (for xgb)
train <- mutate_if(train, is.integer, as.numeric)

# set up data
# full data 
full_variables <- data.matrix(train[,-1]) # with country_destination removed
full_label <- as.numeric(train$country_destination) - 1 # converting to numeric, subtracting 1 (to work with xgb)
full_matrix <- xgb.DMatrix(data = full_variables, label = full_label)

# training data 
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
train_data <- full_variables[train_index, ]
train_label <- full_label[train_index[,1]]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)

# test data 
test_data <- full_variables[-train_index, ]
test_label <- full_label[-train_index[,1]]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)

# ================================================================================================================

# 5-fold cross validation on train data

n_classes <- length(unique(train$country_destination))

parameters <- list("objective" = "multi:softprob",
                   "num_class" = n_classes,
                   eta = 0.3, 
                   gamma = 0, 
                   max_depth = 6, 
                   min_child_weight = 1, 
                   subsample = 0.8, 
                   colsample_bytree = 0.9)
n_round <- 10
cv_fold <- 5

cv_model <- xgb.cv(params = parameters,
                   data = train_matrix,
                   nrounds = n_round,
                   nfold = cv_fold,
                   early_stop_round = 1,
                   verbose = F,
                   maximize = T,
                   prediction = T)

# out of fold predictions 
out_of_fold_p <- data.frame(cv_model$pred) %>% mutate(max_prob = max.col(., ties.method = "last"),
                                                      label = train_label + 1)
head(out_of_fold_p)

# confusion matrix
table(out_of_fold_p$max_prob, out_of_fold_p$label) # only predicting countries 3,5,8,10,12 - mostly predicting 12 (US) (Germany, France, NDF, Other, US)

sum(out_of_fold_p$max_prob == out_of_fold_p$label)/nrow(out_of_fold_p) # 87.6% accuracy

# ==============================================================================================================

# train the full model + test on held out set

full_model <- xgb.train(params = parameters,
                        data = train_matrix,
                        nrounds = n_round)

# Predict hold-out test set
heldout_test_pred <- predict(full_model, newdata = test_matrix)
predictions <- matrix(heldout_test_pred, 
                      nrow = n_classes, 
                      ncol=length(heldout_test_pred)/n_classes) %>% t() %>% data.frame() %>% mutate(label = test_label + 1,
                                                                                                    max_prob = max.col(., "last"))
# not working because not all classes are predicted
# confusionMatrix(factor(predictions$label),
#                 factor(predictions$max_prob),
#                 mode = "everything")

table(predictions$max_prob, predictions$label) # only predicting countries 8, 10, 12

sum(predictions$max_prob == predictions$label) / nrow(predictions) # 87.6% accuracy 

# variable importance
importance <- xgb.importance(colnames(train_matrix), full_model)
head(importance)

# most important firstbook_y.1: gain = 0.98 (improvement in accuracy from firstbook_y.1)
# gain: improvement in accuracy from the feature split on 
# cover: measures relative quantity of observations concerned by a feature
# frequency: counts number of times a feature is used in all generated trees 

first_20 <- importance[1:20,]
xgb.plot.importance(first_20)

# ==============================================================================================================

# fitting model with only the features from xgb.importance output (only 250)

# ==============================================================================================================

new_train <- train %>% select(country_destination, importance$Feature)
# save as csv file to be called in stacking.r
# write.csv(new_train, "xgb_train.csv")

xgb_vars <- colnames(new_train)[-1] # for stacking (244 features)

# set up data

# training data 
train_index1 <- caret::createDataPartition(y = new_train$country_destination, p = 0.70, list = FALSE)
train_data1 <- data.matrix(new_train[train_index1, -1])
train_label1 <- as.numeric(new_train[train_index1, 1]) - 1
train_matrix1 <- xgb.DMatrix(data = train_data1, label = train_label1)

# test data 
test_data1 <- data.matrix(new_train[-train_index1, -1])
test_label1 <- as.numeric(new_train[-train_index1, 1]) - 1
test_matrix1 <- xgb.DMatrix(data = test_data1, label = test_label1)

# 5-fold CV
cv_model1 <- xgb.cv(params = parameters,
                   data = train_matrix1,
                   nrounds = n_round,
                   nfold = cv_fold,
                   early_stop_round = 1,
                   verbose = F,
                   maximize = T,
                   prediction = T)

# out of fold predictions 
out_of_fold_p1 <- data.frame(cv_model1$pred) %>% mutate(max_prob = max.col(., ties.method = "last"),
                                                      label = train_label1 + 1)
head(out_of_fold_p1)

# confusion matrix
table(out_of_fold_p1$max_prob, out_of_fold_p1$label) # only predicting countries 5,8,10,12 

sum(out_of_fold_p1$max_prob == out_of_fold_p1$label)/nrow(out_of_fold_p1) # 87.6% accuracy

# fitting to full train data

full_model1 <- xgb.train(params = parameters,
                        data = train_matrix1,
                        nrounds = n_round)

# Predict hold-out test set
heldout_test_pred1 <- predict(full_model1, newdata = test_matrix1)
predictions1 <- matrix(heldout_test_pred1, 
                      nrow = n_classes, 
                      ncol = length(heldout_test_pred1) / n_classes) %>% t() %>% data.frame() %>% mutate(label = test_label1 + 1,
                                                                                                    max_prob = max.col(., "last"))

table(predictions1$max_prob, predictions1$label) # only predicting countries 8, 10, 12

sum(predictions1$max_prob == predictions1$label) / nrow(predictions1) # 87.6% accuracy 

# variable importance
importance1 <- xgb.importance(colnames(train_matrix1), full_model1)
head(importance1)

first_201 <- importance1[1:20,]
xgb.plot.importance(first_201)

