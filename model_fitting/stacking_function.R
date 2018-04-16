# -------------------------------------------------------------------------------
# 5 fold CV with Stacked Model 
# -------------------------------------------------------------------------------

library(dplyr)
library(magrittr)
library(caret)
library(xgboost)
library(randomForest)

# -----------------------------------------------
set.seed(444)
folds <- caret::createFolds(y = train$country_destination, 
                            k = 5, 
                            list = TRUE, 
                            returnTrain = TRUE) # if returnTrain false, will return test sets

train <- mutate_if(train, is.integer, as.numeric)
train[ ,colnames(train)[colSums(is.na(train)) > 0]] <- -1

# -----------------------------------------------
# Parameters for XGBoost

parameters <- list("objective" = "multi:softprob",
                   "num_class" = 12,
                   eta = 0.3, 
                   max_depth = 8, 
                   min_child_weight = 1, 
                   subsample = 0.8)
n_round <- 10
# -----------------------------------------------

# FUNCTION TO PERFORM STACKING 

cv_stacked <- function(training_folds) {
  
  train_meta <- train[training_folds, ] # set up train and test 
  test_meta <- train[-training_folds, ]
  
  train_meta$fold <- sample(c(1:5), size = nrow(train_meta), prob = rep(0.2, times = 5), replace = TRUE) # creating folds 
  
  train_meta <- cbind(train_meta, M1 = 0, M2 = 0)
  test_meta <- cbind(test, M1 = 0, M2 = 0) 
  
  # XGBoost 
  train_meta$M1[train_meta$fold == 1] <- cv_xgboost(train1 = 2, train2 = 3, train3 = 4, train4 = 5, test = 1, data = train_meta)
  train_meta$M1[train_meta$fold == 2] <- cv_xgboost(train1 = 3, train2 = 4, train3 = 5, train4 = 1, test = 2, data = train_meta)
  train_meta$M1[train_meta$fold == 3] <- cv_xgboost(train1 = 4, train2 = 5, train3 = 1, train4 = 2, test = 3, data = train_meta)
  train_meta$M1[train_meta$fold == 4] <- cv_xgboost(train1 = 5, train2 = 1, train3 = 2, train4 = 3, test = 4, data = train_meta)
  train_meta$M1[train_meta$fold == 5] <- cv_xgboost(train1 = 1, train2 = 2, train3 = 3, train4 = 4, test = 5, data = train_meta)
  
  # random forest 
  train_meta$M2[train_meta$fold == 1] <- cv_rf(train1 = 2, train2 = 3, train3 = 4, train4 = 5, test = 1, data = train_meta)
  train_meta$M2[train_meta$fold == 2] <- cv_rf(train1 = 3, train2 = 4, train3 = 5, train4 = 1, test = 2, data = train_meta)
  train_meta$M2[train_meta$fold == 3] <- cv_rf(train1 = 4, train2 = 5, train3 = 1, train4 = 2, test = 3, data = train_meta)
  train_meta$M2[train_meta$fold == 4] <- cv_rf(train1 = 5, train2 = 1, train3 = 2, train4 = 3, test = 4, data = train_meta)
  train_meta$M2[train_meta$fold == 5] <- cv_rf(train1 = 1, train2 = 2, train3 = 3, train4 = 4, test = 5, data = train_meta)
  
  # -----------------------------------------------
  # Fit xgboost to full training data 
  # -----------------------------------------------
  
  train_xgb <- train_meta %>% select(imp_f_xgb, country_destination) # set up train + test data
  test_xgb <- test_meta %>% select(imp_f_xgb, country_destination)

  full_train <- xgb.DMatrix(data = data.matrix(train_xgb %>% select(-country_destination)), 
                            label = as.numeric(train_xgb$country_destination) - 1)
  full_test <- xgb.DMatrix(data = data.matrix(test_xgb %>% select(-country_destination)), 
                           label = as.numeric(test_xgb$country_destination) - 1)
  # fit model 
  xgb_full <- xgb.train(params = parameters, 
                        data = full_train, 
                        nrounds = n_round)
  
  # predict
  xgb_preds <- predict(xgb_full, newdata = full_test)
  
  # store predictions as df 
  preds_df <- as.data.frame(matrix(xgb_preds, nrow = length(xgb_preds) / 12, 
                                   ncol = 12, 
                                   byrow = TRUE)) %>% mutate(label = as.numeric(test_meta$country_destination),
                                                             max_prob = max.col(., "last"))
  test_meta$M1 <- preds_df$max_prob
  
  # -----------------------------------------------
  # Fit random forest to full training data 
  # -----------------------------------------------
  
  train_meta_rf <- train_meta %>% select(imp_f_rf, country_destination)
  
  rf_full <- randomForest(country_destination ~ ., 
                          data = train_meta_rf, 
                          ntree = 50, 
                          do.trace = 10, 
                          type = "prob")
  
  test_meta_rf <- test_meta %>% select(imp_f_rf, country_destination)
  test_meta$M2 <- predict(rf_full, newdata = test_meta_rf)
  
  # -----------------------------------------------
  # stacking 
  # -----------------------------------------------
  
  train_meta <- mutate_if(train_meta, is.integer, as.numeric)
  test_meta <- mutate_if(test_meta, is.integer, as.numeric)
  
  # set up training data 
  stacked_train <- xgb.DMatrix(data = data.matrix(train_meta %>% select(M1, M2)), 
                               label = as.numeric(train_meta$country_destination) - 1)
  # set up test
  stacked_test <- xgb.DMatrix(data = data.matrix(test_meta %>% select(M1, M2)),
                              label = as.numeric(test_meta$country_destination) - 1)
  # fit xgboost 
  stacked_xgb <- xgb.train(params = parameters, 
                           data = stacked_train, 
                           nrounds = n_round)
  # predict on test 
  stacked_preds <- predict(stacked_xgb, newdata = stacked_test)
  
  # convert to data frame
  stacked_predsdf <- as.data.frame(matrix(stacked_preds, 
                                          nrow = length(stacked_preds) / 12, 
                                          ncol = 12, 
                                          byrow = TRUE)) %>% mutate(label = (getinfo(stacked_test, "label") + 1),
                                                                    max_prob = max.col(., "last"))
  print("Accuracy of stacked model: ")
  (accuracy <- sum(stacked_predsdf$max_prob == stacked_predsdf$label) / length(stacked_predsdf$max_prob))
  
  # confusion matrix 
  print("Confusion matrix of stacked model:")
  (cf_matrix <- table(stacked_predsdf$max_prob, stacked_predsdf$label))
  
  # ncdg metric 
  print("NDCG metric of stacked model:")
  (ndcg <- ndcg5(stacked_preds, stacked_test)) # 0.962
  
  return(list(accuracy, 
              cf_matrix, 
              ndcg, 
              data.frame(train_meta$M1, train_meta$M2), 
              data.frame(test_meta$M1, test_meta$M2)))
  
}

start_time <- Sys.time()
iteration_1 <- cv_stacked(training_folds = folds$Fold1)
iteration_2 <- cv_stacked(training_folds = folds$Fold2)
iteration_3 <- cv_stacked(training_folds = folds$Fold3)
iteration_4 <- cv_stacked(training_folds = folds$Fold4)
iteration_5 <- cv_stacked(training_folds = folds$Fold5)
end_time <- Sys.time()


# -----------------------------------------------------------------------------
# RESULTS OF CV WITH STACKED MODEL
# -----------------------------------------------------------------------------

# average accuracy 
mean(iteration_1[[1]],
     iteration_2[[1]],
     iteration_3[[1]],
     iteration_4[[1]],
     iteration_5[[1]]) # 0.8757594

# average NDCG score
mean(iteration_1[[3]]$value,
     iteration_2[[3]]$value,
     iteration_3[[3]]$value,
     iteration_4[[3]]$value,
     iteration_5[[3]]$value) # 0.9258454

end_time - start_time # 3.242782 hours


