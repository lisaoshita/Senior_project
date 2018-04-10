# =================================================================================================================
# Over/under sampling 
# =================================================================================================================

library(magrittr)
library(dplyr)
library(xgboost)
library(randomForest)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))
train <- train[, -1]
train <- mutate_if(train, is.integer, as.numeric)

# undersample from NDF and US 
# oversample from others 

# perform sampling only on training set 

set.seed(444)

train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)

training <- train[train_index, ]
test <- train[-train_index, ] # don't use this - hold out for stacking.R 

# set up training and test sets
training_s_index <- caret::createDataPartition(y = training$country_destination, p = 0.70, list = FALSE)
training_s <- training[training_s_index, ]
test_s <- training[-training_s_index, ]

# =================================================================================================================

sampled_list <- list()

# oversampling
sampled_list[[1]] <- training_s %>% filter(country_destination == "AU") %>% sample_n(size = 2500, replace = TRUE)
sampled_list[[2]] <- training_s %>% filter(country_destination == "CA") %>% sample_n(size = 2500, replace = TRUE)
sampled_list[[3]] <- training_s %>% filter(country_destination == "DE") %>% sample_n(size = 2500, replace = TRUE)
sampled_list[[4]] <- training_s %>% filter(country_destination == "NL") %>% sample_n(size = 2500, replace = TRUE)
sampled_list[[5]] <- training_s %>% filter(country_destination == "PT") %>% sample_n(size = 2500, replace = TRUE)
sampled_list[[6]] <- training_s %>% filter(country_destination == "ES") %>% sample_n(size = 2500, replace = TRUE)
sampled_list[[7]] <- training_s %>% filter(country_destination == "GB") %>% sample_n(size = 2500, replace = TRUE)
sampled_list[[8]] <- training_s %>% filter(country_destination == "IT") %>% sample_n(size = 2500, replace = TRUE)

# undersampling
sampled_list[[9]] <- training_s %>% filter(country_destination == "NDF") %>% sample_n(size = 25000, replace = TRUE)
sampled_list[[10]] <- training_s %>% filter(country_destination == "US") %>% sample_n(size = 15000, replace = TRUE)

# none
sampled_list[[11]] <- training_s %>% filter(country_destination == "FR") #%>% sample_n(size = 10000, replace = TRUE)
sampled_list[[12]] <- training_s %>% filter(country_destination == "other")

sampled_train <- bind_rows(sampled_list)

# converting NAs in sampled_train and test to -1 (xgb can't handle NAs)
sampled_train[ ,colnames(sampled_train)[colSums(is.na(sampled_train)) > 0]] <- -1
test[ ,colnames(test)[colSums(is.na(test)) > 0]] <- -1

# shuffling order of sampled_data
sampled_train <- sampled_train[sample(nrow(sampled_train)), ] 


# =================================================================================================================
# fit xgboost to new sampled training set 
# =================================================================================================================

# convert to Dmatrix 
sampled_train_d <- xgb.DMatrix(data = data.matrix(sampled_train[ , -1]),
                               label = as.numeric(sampled_train$country_destination) - 1)

# parameters for xgb
parameters <- list("objective" = "multi:softprob",
                   "num_class" = 12,
                   eta = 0.3, 
                   max_depth = 6, 
                   min_child_weight = 1, 
                   subsample = 0.8)

n_round <- 10

# fit to training 
xgb_sampled <- xgb.train(params = parameters, 
                         data = sampled_train_d,
                         nrounds = n_round)

# convert test to Dmatrix
test_s_d <- xgb.DMatrix(data = data.matrix(test_s[, -1]),
                        label = as.numeric(test_s$country_destination) - 1)

# predict 
sampled_preds_xgb <- predict(xgb_sampled, newdata = test_s_d)

# convert predictions to df 
sampled_predsdf_xgb <- as.data.frame(matrix(sampled_preds_xgb, 
                                        nrow = length(sampled_preds_xgb) / 12, 
                                        ncol = 12, 
                                        byrow = TRUE)) %>% mutate(label = getinfo(test_s_d, "label") + 1,
                                                                  max_prob = max.col(., "last"))

# accuracy 
sum(sampled_predsdf_xgb$max_prob == sampled_predsdf_xgb$label) / nrow(sampled_predsdf_xgb) # 0.87

# confusion matrix 
table(sampled_predsdf_xgb$max_prob, sampled_predsdf_xgb$label) # predicting country 8 perfectly 
beepr::beep()

# feature importance
feature_imp_s_xgb <- xgb.importance(feature_names = colnames(sampled_train_d), # 260 features 
                                    model = xgb_sampled)
xgb.plot.importance(feature_imp_s_xgb[1:20])

# ncdg5 metric 
ndcg5(sampled_preds_xgb, test_s_d) # 0.916

# =================================================================================================================
# fit random forest to sampled data 
# =================================================================================================================

rf_model_s <- randomForest(country_destination ~ ., 
                           data = sampled_train, 
                           ntree = 50, 
                           importance = TRUE, 
                           do.trace = 10)

rf_model_s # 88.1% accuracy

rf_pred_s <- predict(rf_model_s, newdata = test_s[,-1])
table(rf_pred_s, test_s$country_destination)
sum(rf_pred_s == test_s$country_destination) / length(rf_pred_s) # 0.8716925

feature_imp_s_rf <- rf_model_s$importance
feature_imp_s_rf <- feature_imp_s_rf[order(-feature_imp_s_rf[, ncol(feature_imp_s_rf)]), ] # decreasing order

head(feature_imp_s_rf)

# =================================================================================================================
# saving features to use in stacking.R
# =================================================================================================================

imp_f_rf <- rownames(feature_imp_s_rf[1:200, ]) 
imp_f_xgb <- feature_imp_s_xgb$Feature[1:200] 


# =================================================================================================================
# stacking with this under/oversampled 
# =================================================================================================================
# stacking with SMOTE data (smoted_train)
train_meta1 <- sampled_train

train_meta1 <- mutate_if(train_meta1, is.integer, as.numeric)


# converting NA to -1 (xgb can't work with NA)
train_meta1[ ,colnames(train_meta1)[colSums(is.na(train_meta1)) > 0]] <- -1
test[ ,colnames(test)[colSums(is.na(test)) > 0]] <- -1

# shuffling order of sampled_train_s
train_meta1 <- train_meta1[sample(nrow(train_meta1)), ] 

# partition training into 5 folds
train_meta1$fold <- sample(c(1:5), size = nrow(train_meta1), prob = rep(0.2, times = 5), replace = TRUE)

# =================================================================================================================

# use train_meta and test_meta1 only in this file 

# creating train_meta and test_meta1, store predictions in M1, M2 
train_meta1 <- cbind(train_meta1, M1 = 0, M2 = 0)
test_meta1 <- cbind(test, M1 = 0, M2 = 0) # test from over_undersampling.R


# =================================================================================================================
# fit xgboost 
# =================================================================================================================

# parameters for xgb
parameters <- list("objective" = "multi:softprob",
                   "num_class" = 12,
                   eta = 0.3, 
                   max_depth = 8, 
                   min_child_weight = 1, 
                   subsample = 0.8)
n_round <- 10


# function to fit xgboost, returns predictions on test set 
cv_xgboost <- function(train1, train2, train3, train4, test) { 
  
  # set up train and test data, include only features from imp_f_xgb
  train <- train_meta1[which(train_meta1$fold == train1 |
                              train_meta1$fold == train2 |
                              train_meta1$fold == train3 | 
                              train_meta1$fold == train4), ] %>% select(imp_f_xgb, country_destination)
  
  test <- train_meta1[which(train_meta1$fold == test), ] %>% select(imp_f_xgb, country_destination)
  
  # convert train and test to Dmatrices
  train_m <- xgb.DMatrix(data = data.matrix(train %>% select(-country_destination)), 
                         label = as.numeric(train$country_destination) - 1)
  
  test_m <- xgb.DMatrix(data = data.matrix(test %>% select(-country_destination)),
                        label = as.numeric(test$country_destination) - 1)
  
  # fit xgboost 
  fit <- xgb.train(params = parameters, 
                   data = train_m, 
                   nrounds = n_round)
  
  # predict on test set 
  preds <- predict(fit, newdata = test_m)
  
  # convert predictions to df 
  preds_df <- as.data.frame(matrix(preds, 
                                   nrow = length(preds) / 12, 
                                   ncol = 12, 
                                   byrow = TRUE)) %>% mutate(label = as.numeric(test[,1]),
                                                             max_prob = max.col(., "last"))
  return(preds_df$max_prob)
  
}

# apply function, store predictions in M1 column 
train_meta1$M1[train_meta1$fold == 1] <- cv_xgboost(train1 = 2, train2 = 3, train3 = 4, train4 = 5, test = 1)
train_meta1$M1[train_meta1$fold == 2] <- cv_xgboost(train1 = 3, train2 = 4, train3 = 5, train4 = 1, test = 2)
train_meta1$M1[train_meta1$fold == 3] <- cv_xgboost(train1 = 4, train2 = 5, train3 = 1, train4 = 2, test = 3)
train_meta1$M1[train_meta1$fold == 4] <- cv_xgboost(train1 = 5, train2 = 1, train3 = 2, train4 = 3, test = 4)
train_meta1$M1[train_meta1$fold == 5] <- cv_xgboost(train1 = 1, train2 = 2, train3 = 3, train4 = 4, test = 5)
beepr::beep()

# train_meta1$M1 <- train_meta1$M1 - 1

# =================================================================================================================
# fit random forest 
# =================================================================================================================

# function to fit rf on training data and predict on test
cv_rf <- function(train1, train2, train3, train4, test) {
  
  # set up train and test sets
  train <- train_meta1[which(train_meta1$fold == train1 |
                              train_meta1$fold == train2 |
                              train_meta1$fold == train3 | 
                              train_meta1$fold == train4), ] %>% select(imp_f_rf, country_destination)
  
  test <- train_meta1[which(train_meta1$fold == test), ] %>% select(imp_f_rf, country_destination)
  
  # fit model 
  rf_model <- randomForest(country_destination ~ ., 
                           data = train, 
                           ntree = 50,
                           do.trace = 10, 
                           type = "prob")
  
  # predict on test 
  pred <- predict(rf_model, newdata = test)
  
  return(pred)
  
}

# store predictions in M2 column 
train_meta1$M2[train_meta1$fold == 1] <- cv_rf(train1 = 2, train2 = 3, train3 = 4, train4 = 5, test = 1)
train_meta1$M2[train_meta1$fold == 2] <- cv_rf(train1 = 3, train2 = 4, train3 = 5, train4 = 1, test = 2)
train_meta1$M2[train_meta1$fold == 3] <- cv_rf(train1 = 4, train2 = 5, train3 = 1, train4 = 2, test = 3)
train_meta1$M2[train_meta1$fold == 4] <- cv_rf(train1 = 5, train2 = 1, train3 = 2, train4 = 3, test = 4)
train_meta1$M2[train_meta1$fold == 5] <- cv_rf(train1 = 1, train2 = 2, train3 = 3, train4 = 4, test = 5)
beepr::beep()

# train_meta1$M2 <- train_meta1$M2 - 1 # 84% accuracy 

# saving as csv
# write.csv(train_meta1, file = "train_stacking.csv")

# =================================================================================================================
# fit each model to full training set, predict on test set, store as predictions in M1, M2
# =================================================================================================================

# xgboost

train1_xgb <- train_meta1 %>% select(imp_f_xgb, country_destination)

test1_xgb <- test_meta1 %>% select(imp_f_xgb, country_destination)

# set up training + test
full_train1 <- xgb.DMatrix(data = data.matrix(train1_xgb %>% select(-country_destination)), 
                          label = as.numeric(train1_xgb$country_destination) - 1)

full_test1 <- xgb.DMatrix(data = data.matrix(test1_xgb %>% select(-country_destination)), 
                         label = as.numeric(test1_xgb$country_destination) - 1)

# fit model 
xgb_full1 <- xgb.train(params = parameters, 
                      data = full_train1, 
                      nrounds = n_round)

# predict
xgb_preds1 <- predict(xgb_full1, newdata = full_test1)

# store predictions as df 
preds_df1 <- as.data.frame(matrix(xgb_preds1, nrow = length(xgb_preds1) / 12, 
                                 ncol = 12, 
                                 byrow = TRUE)) %>% mutate(label = as.numeric(test_meta1$country_destination),
                                                           max_prob = max.col(., "last"))
# checking accuracy
sum(preds_df1$max_prob == preds_df1$label) / nrow(preds_df1) # 87.4% 
table(preds_df1$max_prob, preds_df1$label)

# ndcg5(xgb_preds, full_test) #  0.9208434

# store predictions in column M1 of test meta 
test_meta1$M1 <- preds_df1$max_prob

# ---------------------------------------------------------------------------------------

# random forest 

train_meta1_rf <- train_meta1 %>% select(imp_f_rf, country_destination)

rf_full1 <- randomForest(country_destination ~ ., 
                        data = train_meta1_rf, 
                        ntree = 50, 
                        do.trace = 10, 
                        type = "prob")
beepr::beep()


# predictions (this achieves 98%??)
test_meta1_rf <- test_meta1 %>% select(imp_f_rf, country_destination)

test_meta1$M2 <- predict(rf_full1, newdata = test_meta1_rf) # 0.86

# =================================================================================================================
# stacking - using xgboost as stacker 
# =================================================================================================================

train_meta1$M1 <- train_meta1$M1 - 1
train_meta1$M2 <- train_meta1$M2 - 1
test_meta1$M1 <- test_meta1$M1 - 1
test_meta1$M2 <- as.numeric(test_meta1$M2) - 1


# set up training data 
stacked_train1 <- xgb.DMatrix(data = data.matrix(train_meta1 %>% select(M1, M2)), 
                             label = as.numeric(train_meta1$country_destination) - 1)

# set up test
stacked_test1 <- xgb.DMatrix(data = data.matrix(test_meta1 %>% select(M1, M2)),
                            label = as.numeric(test_meta1$country_destination) - 1)

# fit xgboost 
stacked_xgb1 <- xgb.train(params = parameters, 
                         data = stacked_train1, 
                         nrounds = n_round)

# predict on test 
stacked_preds1 <- predict(stacked_xgb1, newdata = stacked_test1)

# convert to data frame
stacked_predsdf1 <- as.data.frame(matrix(stacked_preds1, 
                                        nrow = length(stacked_preds1) / 12, 
                                        ncol = 12, 
                                        byrow = TRUE)) %>% mutate(label = (getinfo(stacked_test1, "label") + 1),
                                                                  max_prob = max.col(., "last"))

# accuracy - 87.6%
sum(stacked_predsdf1$max_prob == stacked_predsdf1$label) / length(stacked_predsdf1$max_prob) #92% accuracy

# confusion matrix 
table(stacked_predsdf1$max_prob, stacked_predsdf1$label)

# ncdg metric 
ndcg5(stacked_preds1, stacked_test1) 
