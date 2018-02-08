# =====================================================================================
# XGBoost
# =====================================================================================

# working with the train df from feature_engineering file 

# =====================================================================================
# set up data

# full data 
full_variables <- data.matrix(train[,-1]) 
full_label <- as.numeric(train$country_destination) - 1
full_matrix <- xgb.DMatrix(data.matrix(train), label = full_label)

# training data 
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
train_data <- full_variables[train_index, ]
train_label <- full_label[train_index[,1]]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)

# test data 
test_data <- full_variables[-train_index, ]
test_label <- full_label[-train_index[,1]]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)

# =====================================================================================

# 5-fold cross validation on train data

n_classes <- length(unique(train$country_destination))

parameters <- list("objective" = "multi:softprob",
                   "num_class" = n_classes,
                   eta = 0.3, 
                   gamma = 0, 
                   max_depth = 6, 
                   min_child_weight = 1, 
                   subsample = 1, 
                   colsample_bytree = 1)
n_round <- 2
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
out_of_fold_p <- data.frame(cv_model$pred) %>% 
  mutate(max_prob = max.col(., ties.method = "last"),
         label = train_label + 1)
head(out_of_fold_p)

# confusion matrix
confusionMatrix(factor(out_of_fold_p$label), 
                factor(out_of_fold_p$max_prob),
                mode = "everything")
# 100% accuracy??

# =====================================================================================

# train the full model + test on held out set

full_model <- xgb.train(params = parameters,
                        data = train_matrix,
                        nrounds = n_round)

# Predict hold-out test set
heldout_test_pred <- predict(full_model, newdata = test_matrix)
test_prediction <- matrix(heldout_test_pred, nrow = n_classes,
                              ncol=length(heldout_test_pred)/n_classes) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))

confusionMatrix(factor(test_prediction$label),
                factor(test_prediction$max_prob),
                mode = "everything")

