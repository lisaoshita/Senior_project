# =================================================================================================================
# XGBoost
# =================================================================================================================
library(xgboost)
library(dplyr)
library(magrittr)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

# removing row with NA for country destination 
train <- train[-which(is.na(train$country_destination)), ]
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
table(out_of_fold_p$max_prob, out_of_fold_p$label) # only predicting countries 5,8,10,12 - mostly predicting 12 (US) (France, NDF, Other, US)

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

table(predictions$max_prob, predictions$label) # only predicting countries 5, 8, 12

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

