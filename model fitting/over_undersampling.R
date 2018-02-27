# =================================================================================================================
# Overssampling and undersampling 
# =================================================================================================================

library(magrittr)
library(dplyr)
library(xgboost)

prop.table(table(train$country_destination)) * 100

# undersample from NDF and US 
# oversample from others 

# perform sampling on training set 
set.seed(444)
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
training <- train[train_index, ]
test <- train[-train_index, ]

# function to sample each country up or down to 10,000
sample_train <- function(country, replace) {
  
  dat <- training %>% filter(country_destination == country) %>% sample_n(size = 10000, replace = replace)
  return(dat)
  
}

countries <- as.character(unique(train$country_destination))
replace_vals <- c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
sampled_train <- purrr::map2_dfr(countries, replace_vals, ~sample_train(.x, .y))

# =================================================================================================================

# fit xgboost to new sampled training set 

# training data 
sampled_train_d <- xgb.DMatrix(data = data.matrix(sampled_train[ , -1]),
                               label = as.numeric(sampled_train$country_destination) - 1)

# parameters for xgb
parameters <- list("objective" = "multi:softprob",
                   "num_class" = 12,
                   eta = 0.3, 
                   gamma = 0, 
                   max_depth = 6, 
                   min_child_weight = 1, 
                   subsample = 0.8, 
                   colsample_bytree = 0.9)

n_round <- 10

# fit to training 
xgb_sampled <- xgb.train(params = parameters, 
                         data = sampled_train_d,
                         nrounds = n_round)

# set up test data for predictions 
test_s <- xgb.DMatrix(data = data.matrix(test[, -1]),
                    label = as.numeric(test$country_destination) - 1)

sampled_preds <- predict(xgb_sampled, newdata = test_s)

# convert predictions to df 
sampled_predsdf <- as.data.frame(matrix(sampled_preds, 
                                        nrow = length(sampled_preds) / 12, 
                                        ncol = 12, 
                                        byrow = TRUE)) %>% mutate(label = as.numeric(test[,1]),
                                                                  max_prob = max.col(., "last"))

# accuracy 
sum(sampled_predsdf$max_prob == sampled_predsdf$label) / nrow(sampled_predsdf) # 63% accuracy 

# confusion matrix 
table(sampled_predsdf$max_prob, sampled_predsdf$label) # predicting country 8 perfectly 
