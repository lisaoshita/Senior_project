# =====================================================================================
# XGBoost
# =====================================================================================

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

# removing row with NA for country destination 
train <- train[-which(is.na(train$country_destination)), ]

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
# 100% accuracy

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
full_s <- data.matrix(train[s, -1]) # 53362 observations, with country_destination removed
full_label_s <- as.numeric(train$country_destination[s]) - 1

# removing row 

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

rounds    <- 5
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
                                                       label = train_s_lab + 1)
head(OOF_prediction)

# confusion matrix
# not working because xgboost only predicts countries 8, 10, 12
# caret::confusionMatrix(factor(OOF_prediction$label),
#                        factor(OOF_prediction$max_prob),
#                        mode = "everything")

table(OOF_prediction$max_prob, OOF_prediction$label)

# function to match number with country
get_countries <- function(predict) {
  predict[predict == 1] <- "AU"; predict[predict == 2] <- "CA"
  predict[predict == 3] <- "DE"; predict[predict == 4] <- "ES"
  predict[predict == 5] <- "FR"; predict[predict == 6] <- "GB"
  predict[predict == 7] <- "IT"; predict[predict == 8] <- "NDF"
  predict[predict == 9] <- "NL"; predict[predict == 10] <- "other"
  predict[predict == 11] <- "PT"; predict[predict == 12] <- "US"
  return(predict)
}

actual <- get_countries(OOF_prediction$label)
preds <- get_countries(OOF_prediction$max_prob)
sum(actual == preds) / nrow(OOF_prediction) # 0.874 accuracy 

# ---------------------------------------------------------------------------

# fitting to full train data 

fit <- xgboost(params = params, data = train_s_m, nrounds = rounds)

# Predict hold-out test set
test_p <- predict(fit, newdata = test_s_m)
test_p <- matrix(test_p, nrow = classes,
                 ncol=length(test_p) / classes) %>% t() %>% data.frame() %>% mutate(label = test_s_lab + 1,
                                                                                    max_prob = max.col(., "last"))
# confusion matrix of test set
# confusionMatrix(factor(test_p$label),
#                 factor(test_p$max_prob),
#                 mode = "everything") 

table(test_p$max_prob, test_p$label) # this model only predicts countries 8 and 12 
actual1 <- get_countries(test_p$label)
preds1 <- get_countries(test_p$max_prob)
sum(actual1 == preds1) / nrow(test_p) #  0.8767491 accuracy 
