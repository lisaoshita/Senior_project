# ========================================================================================
# Generalized stacking

# from: http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
# ========================================================================================

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

# partition data into 5 test folds 
training$fold <- sample(c(1:5), size = nrow(training), prob = rep(0.2, times = 5), replace = TRUE)

# creating train_meta and test_meta
train_meta <- cbind(training, M1 = 0, M2 = 0)
test_meta <- cbind(test, M1 = 0, M2 = 0)

# fit model to training fold, and predict on test fold (hold out one fold, and combine other folds to train)
# train on 2,3,4,5, test on 1
# train on 3,4,5,1, test on 2...


# ========================================================================================
# xgboost (put predictions in M1)
train_set <- train_meta[which(train_meta$fold == 2 | 
                                train_meta$fold == 3 | 
                                train_meta$fold == 4 | 
                                train_meta$fold == 5), ]
test_set <- train_meta[which(train_meta$fold == 1), ]

exclude <- c(which(colnames(train_set) == "M1"), 
             which(colnames(train_set) == "M2"), 
             which(colnames(train_set) == "fold"))

train_m <- xgb.DMatrix(data = data.matrix(train_set[, -c(1, exclude)]), 
                       label = as.numeric(train_set[,1]) - 1)
test_m <- xgb.DMatrix(data = data.matrix(test_set[, -c(1, exclude)]), 
                      label = as.numeric(test_set[,1]) - 1)

parameters <- list("objective" = "multi:softprob",
                   "num_class" = 12,
                   eta = 0.3, 
                   gamma = 0, 
                   max_depth = 6, 
                   min_child_weight = 1, 
                   subsample = 0.8, 
                   colsample_bytree = 0.9)
n_round <- 10

xgb_fit <- xgb.train(params = parameters,
                     data = train_m,
                     nrounds = n_round)

test_pred <- predict(xgb_fit, newdata = test_m)

predictions <- as.data.frame(matrix(test_pred, nrow = length(test_pred) / 12, ncol = 12, byrow = TRUE)) %>% mutate(label = as.numeric(test_set[,1]),
                                                                                                                   max_prob = max.col(., "last"))

train_meta$M1[train_meta$fold == 1] <- predictions$max_prob

# function to perform CV for xgboost, returns predictions on test set 
cv_xgboost <- function(train1, train2, train3, train4, test) { 
  
  # set up train and test data 
  train <- train_meta[which(train_meta$fold == train1 |
                              train_meta$fold == train2 |
                              train_meta$fold == train3 | 
                              train_meta$fold == train4), ]
  test <- train_meta[which(train_meta$fold == 1), ]
  
  # exclude M1, M2 and fold columns of train and test sets
  exclude <- c(which(colnames(train) == "M1"), 
               which(colnames(train) == "M2"), 
               which(colnames(train) == "fold"))
  
  # convert to Dmatrix
  train_m <- xgb.DMatrix(data = data.matrix(train[, -c(1, exclude)]), 
                         label = as.numeric(train[,1]) - 1)
  test_m <- xgb.DMatrix(data = data.matrix(test[, -c(1, exclude)]), 
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

y <- cv_xgboost(train1 = 2, train2 = 3, train3 = 4, train4 = 5, test = 1)






















