# ========================================================================================
# SMOTE
# ========================================================================================
# Synthetic Minority Oversampling Technique -  k nearest neighbors randomly chosen

# load packages
library(smotefamily)
library(magrittr)
library(dplyr)
library(FNN) # SMOTE won't work without this package
library(randomForest)
library(xgboost)

# ========================================================================================
vars <- c(imp_f_rf, imp_f_xgb)
vars <- unique(vars)

# function to perform SMOTE 
perform_smote <- function(country, up_size) {
  
  df <- training %>% 
    filter(country_destination == country |
             country_destination == "NDF") %>% 
    select(country_destination, vars)
  
  smoted <- SMOTE(df[, -1], 
                  target = as.numeric(df$country_destination),
                  K = 5,
                  dup_size = up_size)
  
  syn_dat <- smoted$syn_data
  
  syn_dat$class <- as.numeric(syn_dat$class)
  
  return(syn_dat)
  
}
# ========================================================================================

table(training$country_destination)

smote_list <- list()

# performing SMOTE on under-represented countries
smote_list[[1]] <- perform_smote(country = "AU", up_size = 15) # 5670 observations
smote_list[[2]] <- perform_smote(country = "CA", up_size = 5)  # 5000
smote_list[[3]] <- perform_smote(country = "DE", up_size = 8)  # 5944
smote_list[[4]] <- perform_smote(country = "ES", up_size = 4)  # 6300
smote_list[[5]] <- perform_smote(country = "FR", up_size = 2)  # 7034
smote_list[[6]] <- perform_smote(country = "GB", up_size = 4)  # 6508
smote_list[[7]] <- perform_smote(country = "IT", up_size = 3)  # 5955
smote_list[[8]] <- perform_smote(country = "NL", up_size = 10) # 5340
smote_list[[9]] <- perform_smote(country = "PT", up_size = 35) # 5320

# leaving class-other alone + undersample from NDF and US
smote_list[[10]] <- training %>% filter(country_destination == "other") %>% select(vars) %>% mutate(class = 10) 
smote_list[[11]] <- training %>% filter(country_destination == "NDF") %>% sample_n(size = 35000, replace = FALSE) %>% select(vars) %>% mutate(class = 8)
smote_list[[12]] <- training %>% filter(country_destination == "US") %>% sample_n(size = 20000, replace = FALSE) %>% select(vars) %>% mutate(class = 12)  

smoted_train <- bind_rows(smote_list)
smoted_train <- smoted_train[sample(nrow(smoted_train)), ]
beepr::beep()

# ========================================================================================
# fit random forest to SMOTEd data 
# ========================================================================================

smoted_train_rf <- smoted_train %>% select(imp_f_rf, class)
smoted_train_rf$class <- as.factor(smoted_train_rf$class)

rf_full <- randomForest(class ~ ., 
                        data = smoted_train_rf, 
                        ntree = 50, 
                        do.trace = 10, 
                        type = "prob")
beepr::beep()

rf_test_df <- test %>% select(imp_f_rf)

rf_preds <- predict(rf_full, newdata = rf_test_df)

table(rf_preds, as.numeric(test$country_destination))

sum(rf_preds == as.numeric(test$country_destination))/length(rf_preds) # 87% accuracy 

table(rf_preds) # still only predicting a few of the countries 

# ========================================================================================
# fit XGB to SMOTEd data 
# ========================================================================================

smoted_train_d <- xgb.DMatrix(data = data.matrix(smoted_train %>% select(imp_f_xgb, -class)),
                              label = as.numeric(smoted_train$class) - 1)

parameters <- list("objective" = "multi:softprob",
                   "num_class" = 12,
                   eta = 0.3, 
                   max_depth = 8, 
                   min_child_weight = 1, 
                   subsample = 0.8)
n_round <- 10


smoted_xgb <- xgb.train(params = parameters, 
                 data = smoted_train_d, 
                 nrounds = n_round)
beepr::beep()

smoted_test_d <- xgb.DMatrix(data = data.matrix(test %>% select(imp_f_xgb)),
                             label = as.numeric(test$country_destination) - 1)

smoted_preds <- predict(smoted_xgb, newdata = smoted_test_d)

smoted_preds_df <- as.data.frame(matrix(smoted_preds, 
                                         nrow = length(smoted_preds) / 12, 
                                         ncol = 12, 
                                         byrow = TRUE)) %>% mutate(label = as.numeric(test$country_destination),
                                                                   max_prob = max.col(., "last"))

# confusion matrix
table(smoted_preds_df$max_prob, smoted_preds_df$label) # only predicting 3,6,8,9,10,11,12 

# accuracy 
sum(smoted_preds_df$max_prob == smoted_preds_df$label) / nrow(smoted_preds_df) # 0.8752753
beepr::beep()
















