# =================================================================================================================
# Generalized stacking

# from: http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
# =================================================================================================================

library(dplyr)
library(caret)
library(xgboost)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

train <- train[, -1]
train <- mutate_if(train, is.integer, as.numeric)

set.seed(444)

# creating training and test sets 
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
training <- train[train_index, ]
test <- train[-train_index, ]

# partition data into 5 folds 
training$fold <- sample(c(1:5), size = nrow(training), prob = rep(0.2, times = 5), replace = TRUE)

# creating train_meta and test_meta, store predictions in M1, M2 
train_meta <- cbind(training, M1 = 0, M2 = 0)
test_meta <- cbind(test, M1 = 0, M2 = 0)

# fit model to training fold, predict on test fold (hold out one fold, and combine other folds to train)
# train on 2,3,4,5, test on 1
# train on 3,4,5,1, test on 2...


# ========================================================================================

# xgboost 

# parameters for xgb
parameters <- list("objective" = "multi:softprob",
                   "num_class" = n_classes,
                   eta = 0.3, 
                   gamma = 0, 
                   max_depth = 6, 
                   min_child_weight = 1, 
                   subsample = 0.8, 
                   colsample_bytree = 0.9)

n_round <- 10


# function to fit xgboost, returns predictions on test set 
cv_xgboost <- function(train1, train2, train3, train4, test) { 
  
  # set up train and test data 
  train <- train_meta[which(train_meta$fold == train1 |
                              train_meta$fold == train2 |
                              train_meta$fold == train3 | 
                              train_meta$fold == train4), ]
  
  test <- train_meta[which(train_meta$fold == test), ]
  
  # convert to Dmatrix
  train_m <- xgb.DMatrix(data = data.matrix(train[, which(colnames(train) %in% xgb_vars & # xgb_vars from xg_boost.R
                                                            colnames(train) != rebus::or("M1", "M2", "fold"))]), 
                         label = as.numeric(train[,1]) - 1)
  
  test_m <- xgb.DMatrix(data = data.matrix(test[, which(colnames(train) %in% xgb_vars & 
                                                           colnames(train) != rebus::or("M1", "M2", "fold"))]), 
                        label = as.numeric(test[,1]) - 1)
  
  # fit xgboost 
  fit <- xgb.train(params = parameters, 
                   data = train_m, 
                   nrounds = n_round)
  
  # predict on test set
  preds <- predict(fit, newdata = test_m)
  
  # convert predictions to df 
  preds_df <- as.data.frame(matrix(preds, nrow = length(preds) / 12, ncol = 12, byrow = TRUE)) %>% mutate(label = as.numeric(test[,1]),
                                                                                                          max_prob = max.col(., "last"))
  return(preds_df$max_prob)
  
  }


# store predictions in M1 column 
train_meta$M1[train_meta$fold == 1] <- cv_xgboost(train1 = 2, train2 = 3, train3 = 4, train4 = 5, test = 1)
train_meta$M1[train_meta$fold == 2] <- cv_xgboost(train1 = 3, train2 = 4, train3 = 5, train4 = 1, test = 2)
train_meta$M1[train_meta$fold == 3] <- cv_xgboost(train1 = 4, train2 = 5, train3 = 1, train4 = 2, test = 3)
train_meta$M1[train_meta$fold == 4] <- cv_xgboost(train1 = 5, train2 = 1, train3 = 2, train4 = 3, test = 4)
train_meta$M1[train_meta$fold == 5] <- cv_xgboost(train1 = 1, train2 = 2, train3 = 3, train4 = 4, test = 5)

# ========================================================================================

# random forest 

# function to fit rf on training data and predict on test
cv_rf <- function(train1, train2, train3, train4, test) {
  
  # set up train and test data 
  train <- train_meta[which(train_meta$fold == train1 |
                              train_meta$fold == train2 |
                              train_meta$fold == train3 | 
                              train_meta$fold == train4), 
                      c(1, which(colnames(train_meta) %in% rf_vars & colnames(train_meta) != rebus::or("M1", "M2", "fold")))]
  
  test <- train_meta[which(train_meta$fold == test), 
                     c(1, which(colnames(train_meta) %in% rf_vars & colnames(train_meta) != rebus::or("M1", "M2", "fold")))]
  
  # fit model 
  rf_model <- randomForest(country_destination ~ ., data = train, ntree = 50, importance = TRUE, do.trace = 10, type = "prob")
  
  # predict on test 
  pred <- predict(rf_model, newdata = test)
  
  return(pred)
  
}

# store predictions in M2 column 
train_meta$M2[train_meta$fold == 1] <- cv_rf(train1 = 2, train2 = 3, train3 = 4, train4 = 5, test = 1)
train_meta$M2[train_meta$fold == 2] <- cv_rf(train1 = 3, train2 = 4, train3 = 5, train4 = 1, test = 2)
train_meta$M2[train_meta$fold == 3] <- cv_rf(train1 = 4, train2 = 5, train3 = 1, train4 = 2, test = 3)
train_meta$M2[train_meta$fold == 4] <- cv_rf(train1 = 5, train2 = 1, train3 = 2, train4 = 3, test = 4)
train_meta$M2[train_meta$fold == 5] <- cv_rf(train1 = 1, train2 = 2, train3 = 3, train4 = 4, test = 5)

# saving as csv
# write.csv(train_meta, file = "train_stacking.csv")

# ========================================================================================

# fit each model to full training set, predict on test set, store as predictions in M1, M2

# xgboost

# set up training + test
full_train <- xgb.DMatrix(data = data.matrix(training[ , -c(1, which(colnames(train) == "fold"))]), 
                          label = as.numeric(training[ , 1]) - 1)
full_test <- xgb.DMatrix(data = data.matrix(test[ , -c("country_destination")]), 
                         label= as.numeric(test[ , 1]) - 1)

# fit model 
xgb_full <- xgb.train(params = parameters, 
                      data = full_train, 
                      nrounds = n_round)

# predictions 
xgb_preds <- predict(xgb_full, newdata = full_test)
preds_df <- as.data.frame(matrix(xgb_preds, nrow = length(xgb_preds) / 12, ncol = 12, byrow = TRUE)) %>% mutate(label = as.numeric(test[,1]),
                                                                                                        max_prob = max.col(., "last"))
# store predictions in column M1 of test meta 
test_meta$M1 <- preds_df$max_prob

# ---------------------------------------------------------------------------------------

# random forest 

rf_full <- randomForest(country_destination ~ ., 
                        data = training[,-which(colnames(training) == "fold")], 
                        ntree = 50, 
                        importance = TRUE, 
                        do.trace = 10, 
                        type = "prob")

# predictions
test_meta$M2 <- predict(rf_full, newdata = test[ , -1])

test_meta <- mutate_if(test_meta, is.integer, as.numeric)

  
# ========================================================================================

# stacking 
# using xgboost as stacker 

# set up training data 
stacked_train <- xgb.DMatrix(data = data.matrix(train_meta[ , c("M1", "M2")]), 
                             label = as.numeric(train_meta[ , 1]) - 1)

# set up test
stacked_test <- xgb.DMatrix(data = data.matrix(test_meta[, c("M1", "M2")]),
                            label = as.numeric(test_meta[ , 1]) - 1)

# fit xgboost 
stacked_xgb <- xgb.train(params = parameters, 
                         data = stacked_train, 
                         nrounds = n_round)

# predict on test 
stacked_preds <- predict(stacked_xgb, newdata = stacked_test)

# convert to data frame
stacked_predsdf <- as.data.frame(matrix(stacked_preds, nrow = length(stacked_preds) / 12, ncol = 12, byrow = TRUE)) %>% 
  mutate(label = as.numeric(test_meta[,1]),
         max_prob = max.col(., "last"))

# accuracy - 87.6%
sum(stacked_predsdf$max_prob == stacked_predsdf$label) / nrow(stacked_predsdf)

# confusion matrix 
table(stacked_predsdf$max_prob, stacked_predsdf$label)

