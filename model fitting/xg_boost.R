# =====================================================================================
# XGBoost
# =====================================================================================

# =====================================================================================
# Code from: https://rpubs.com/mharris/multiclass_xgboost
# =====================================================================================

library("xgboost")  
library("archdata") 
library("caret")   
library("dplyr") 
library("magrittr")

set.seed(717)
data(RBGlass1)
dat <- RBGlass1 

# converting response to numeric + adding another class
dat$Site <- as.numeric(dat$Site)
dat_add <- dat[which(dat$Site == 1),] %>%
  rowwise() %>%
  mutate_all(funs(./10 + rnorm(1,.,.*0.1))) %>%
  mutate_all(funs(round(.,2))) %>%
  mutate(Site = 3)
dat <- rbind(dat, dat_add) %>%
  mutate(Site = Site - 1)

# creating test/train splits
in_train <- createDataPartition(y = dat$Site, p = 0.75, list = FALSE)

# full data set
full_vars <- as.matrix(dat[,-1]) # removed site
full_lab <- dat[, "Site"]
full_m <- xgb.DMatrix(data = full_vars, label = full_lab)

# train 
train_dat <- full_vars[in_train, ]
train_lab <- full_lab[in_train[,1]]
train_m <- xgb.DMatrix(data = train_dat, label = train_lab)

# test 
test_dat <- full_vars[-in_train,]
test_lab <- full_lab[-in_train[,1]]
test_m <- xgb.DMatrix(data = test_dat, label = test_lab)

# --------------------------------------------------------------------
# k-fold CV

numberOfClasses <- length(unique(dat$Site))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)

nround    <- 50 
cv.nfold  <- 5

# fit 5-fold CV 50 times and save out of fold predictions
cv <- xgb.cv(params = xgb_params,
             data = train_m, 
             nrounds = nround,
             nfold = cv.nfold,
             verbose = FALSE,
             prediction = TRUE)

# using max.col to assign a class
OOF_prediction <- data.frame(cv$pred) %>% mutate(max_prob = max.col(., ties.method = "last"),
                                                 label = train_lab + 1)
head(OOF_prediction)

# confusion matrix
confusionMatrix(factor(OOF_prediction$label), 
                factor(OOF_prediction$max_prob),
                mode = "everything")
# 85% accuracy 

# --------------------------------------------------------------------

# fitting to full train data 
bst_model <- xgb.train(params = xgb_params,
                       data = train_m,
                       nrounds = nround)
# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_m)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>% t() %>% data.frame() %>% mutate(label = test_lab + 1,
                                                                                                      max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$label),
                factor(test_prediction$max_prob),
                mode = "everything")

# achieved 73% accuracy 


# =====================================================================================
# Airbnb data: working with the train df from feature_engineering file 
# =====================================================================================
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
out_of_fold_p <- data.frame(cv_model$pred) %>% mutate(max_prob = max.col(., ties.method = "last"),
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
predictions <- matrix(heldout_test_pred, 
                      nrow = n_classes, 
                      ncol=length(heldout_test_pred)/n_classes) %>% t() %>% data.frame() %>% mutate(label = test_label + 1,
                                                                                                    max_prob = max.col(., "last"))

confusionMatrix(factor(predictions$label),
                factor(predictions$max_prob),
                mode = "everything")
# 100 % 

# =====================================================================================

# working with only a sample -------------------------------------------------
s <- sample(1:nrow(train), nrow(train)/4) # try with a quarter of the data
full_s <- data.matrix(train[s, -1]) # 53362 observations 
full_label_s <- as.numeric(train$country_destination[s]) - 1

# train
t <- caret::createDataPartition(y = full_label_s, p = 0.70, list = FALSE)
train_s <- full_s[t, ]
train_s_lab <- full_label_s[t]
train_s_m <- xgb.DMatrix(data = train_s, label = train_s_lab)

# test 
test_s <- full_s[-t, ]
test_s_lab <- full_label_s[-t]
test_s_m <- xgb.DMatrix(data = test_s, label = test_s_lab)
# ---------------------------------------------------------------------------

# 5 fold CV

classes <- length(unique(full_label_s))

params <- list("objective" = "multi:softprob",
                   "num_class" = classes,
                   eta = 0.3, 
                   max_depth = 6)

rounds    <- 50 
folds  <- 5

# fit 5-fold CV 50 times and save out of fold predictions
cv_model <- xgb.cv(params = params,
                   data = train_s_m, 
                   nrounds = rounds,
                   nfold = folds,
                   verbose = FALSE,
                   prediction = TRUE)

# using max.col to assign a class
OOF_prediction <- data.frame(cv_model$pred) %>% mutate(max_prob = max.col(., ties.method = "last"),
                                                       label = train_lab + 1)
head(OOF_prediction)

# confusion matrix
confusionMatrix(factor(OOF_prediction$label), 
                factor(OOF_prediction$max_prob),
                mode = "everything")

# ---------------------------------------------------------------------------

# fitting to full train data 

fit <- xgb.train(params = params, data = train_s_m, nrounds = rounds)

# Predict hold-out test set
test_p <- predict(fit, newdata = test_s_m)
test_p <- matrix(test_p, nrow = classes,
                 ncol=length(test_p) / classes) %>% t() %>% data.frame() %>% mutate(label = test_s_lab + 1,
                                                                                    max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_p$label),
                factor(test_p$max_prob),
                mode = "everything") 
