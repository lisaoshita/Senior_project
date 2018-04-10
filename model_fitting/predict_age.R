# =================================================================================================================
# Predict Age
# =================================================================================================================

library(class)
library(magrittr)
library(dplyr)
library(xgboost)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

# set up train + test sets 
# training set contains all rows with a valid age
age_train <- train %>% 
                filter(age_clean != -1) %>% 
                select(-c(country_destination, starts_with("age_bucket")))

# test set contains all rows with -1 as age 
age_test <- train %>% 
                filter(age_clean == -1) %>% 
                select(-c(country_destination, starts_with("age_bucket")))


set.seed(444)

# split training set into train + test
train_index <- caret::createDataPartition(y = age_train$country_destination, p = 0.70, list = FALSE)

age_training <- age_train[train_index, ]

age_testing <- age_train[-train_index, ]

# convert to DMatrix
age_train_d <- xgb.DMatrix(data = data.matrix(age_training[, -which(colnames(age_training) == "age_clean")]),
                           label = age_training$age_clean)

age_test_d <- xgb.DMatrix(data = data.matrix(age_testing[, -which(colnames(age_training) == "age_clean")]),
                          label = age_testing$age_clean)

# =================================================================================================================

# fit xgboost to predict age 

# parameters 
param <- list("objective" = "reg:linear",
              "eta" = 0.3,
              "max_depth" = 10,
              "subsample" = 0.7,
              "colsample_bytree" = 0.3,
              "alpha" = 1.0)

# fit model 
age_xgb <- xgb.train(params = param,
                     data = age_train_d,
                     nrounds = 10)

# predict on heldout test set
age_predict <- predict(age_xgb, newdata = age_test_d)


age_predsdf <- data.frame(predictions = age_predict, 
                          real = age_testing$age_clean) %>% mutate(difference = predictions - real)

# mean square error 
sqrt(sum((x$predictions - x$real)**2))

# largest difference between prediction and actual is 82 years 


# =================================================================================================================


# fit to entire training set and fill in NAs in age_clean 

# set up as Dmatrix
age_traind <- xgb.DMatrix(data = data.matrix(age_train %>% select(-c(age_clean))),
                          label = age_train$age_clean)

# fit xgboost
age_xgb_train <- xgb.train(params = param,
                           data = age_traind,
                           nrounds = 10)

# predict on test set 

age_test$age_clean <- NA
age_testd <- xgb.DMatrix(data = data.matrix(age_test %>% select(-c(age_clean))),
                         label = age_test$age_clean)

# save predictions in train

train$age_clean[train$age_clean == -1] <- predict(age_xgb_train, newdata = age_testd)














 