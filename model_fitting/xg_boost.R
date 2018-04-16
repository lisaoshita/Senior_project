# =================================================================================================================
# XGBoost
# =================================================================================================================

library(xgboost)
library(dplyr)
library(magrittr)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

# removing first column 
train <- train[,-1]
# converting all column that are integers to numeric (for xgb)
train <- mutate_if(train, is.integer, as.numeric)

# set up data
# training data 
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
training <- train[train_index, ]
train_matrix <- xgb.DMatrix(data = data.matrix(training[, -1]), 
                            label = as.numeric(training$country_destination) - 1)

# test data 
test <- train[-train_index, ]
test_matrix <- xgb.DMatrix(data = data.matrix(test[, -1]), 
                           label = as.numeric(test$country_destination) - 1)

# ================================================================================================================

# 5-fold cross validation on train data

n_classes <- length(unique(train$country_destination))

parameters <- list("objective" = "multi:softprob",
                   "num_class" = 12,
                   eta = 0.3, 
                   max_depth = 8, 
                   min_child_weight = 1, 
                   subsample = 0.8)
n_round <- 10
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
table(out_of_fold_p$max_prob, out_of_fold_p$label) # only predicting countries 3,5,8,10,12 - mostly predicting 12 (US) (Germany, France, NDF, Other, US)

sum(out_of_fold_p$max_prob == out_of_fold_p$label)/nrow(out_of_fold_p) # 87.6% accuracy

# ==============================================================================================================

# train the full model + test on held out set

full_model <- xgb.train(params = parameters,
                        data = train_matrix,
                        nrounds = n_round)
beepr::beep()

# Predict hold-out test set
heldout_test_pred <- predict(full_model, newdata = test_matrix)
predictions <- matrix(heldout_test_pred, 
                      nrow = 12, 
                      ncol=length(heldout_test_pred)/n_classes) %>% t() %>% data.frame() %>% mutate(label = test_label + 1,
                                                                                                    max_prob = max.col(., "last"))
# not working because not all classes are predicted
# confusionMatrix(factor(predictions$label),
#                 factor(predictions$max_prob),
#                 mode = "everything")

ndcg5(heldout_test_pred, test_matrix) # 0.927 

table(predictions$max_prob, predictions$label) # only predicting countries 8, 10, 12

sum(predictions$max_prob == predictions$label) / nrow(predictions) # 87.6% accuracy 

# variable importance
importance <- xgb.importance(colnames(train_matrix), full_model)
head(importance)

# build model off of this 
imp_f_xgb <- importance$Feature[1:200]

# most important firstbook_y.1: gain = 0.98 (improvement in accuracy from firstbook_y.1)
# gain: improvement in accuracy from the feature split on 
# cover: measures relative quantity of observations concerned by a feature
# frequency: counts number of times a feature is used in all generated trees 

first_20 <- importance[1:20,]
xgb.plot.importance(first_20)

# ==============================================================================================================

# fitting model with only the features from xgb.importance output (only 250)

# ==============================================================================================================

new_train <- train %>% select(country_destination, importance$Feature)
# save as csv file to be called in stacking.r
# write.csv(new_train, "xgb_train.csv")

xgb_vars <- colnames(new_train)[-1] # for stacking (244 features)

# set up data

# training data 
train_index1 <- caret::createDataPartition(y = new_train$country_destination, p = 0.70, list = FALSE)
train_data1 <- data.matrix(new_train[train_index1, -1])
train_label1 <- as.numeric(new_train[train_index1, 1]) - 1
train_matrix1 <- xgb.DMatrix(data = train_data1, label = train_label1)

# test data 
test_data1 <- data.matrix(new_train[-train_index1, -1])
test_label1 <- as.numeric(new_train[-train_index1, 1]) - 1
test_matrix1 <- xgb.DMatrix(data = test_data1, label = test_label1)

# 5-fold CV
cv_model1 <- xgb.cv(params = parameters,
                   data = train_matrix1,
                   nrounds = n_round,
                   nfold = cv_fold,
                   early_stop_round = 1,
                   verbose = F,
                   maximize = T,
                   prediction = T)

# out of fold predictions 
out_of_fold_p1 <- data.frame(cv_model1$pred) %>% mutate(max_prob = max.col(., ties.method = "last"),
                                                      label = train_label1 + 1)
head(out_of_fold_p1)

# confusion matrix
table(out_of_fold_p1$max_prob, out_of_fold_p1$label) # only predicting countries 5,8,10,12 

sum(out_of_fold_p1$max_prob == out_of_fold_p1$label)/nrow(out_of_fold_p1) # 87.6% accuracy

# fitting to full train data

full_model1 <- xgb.train(params = parameters,
                        data = train_matrix1,
                        nrounds = n_round)

# Predict hold-out test set
heldout_test_pred1 <- predict(full_model1, newdata = test_matrix1)
predictions1 <- matrix(heldout_test_pred1, 
                      nrow = n_classes, 
                      ncol = length(heldout_test_pred1) / n_classes) %>% t() %>% data.frame() %>% mutate(label = test_label1 + 1,
                                                                                                    max_prob = max.col(., "last"))

table(predictions1$max_prob, predictions1$label) # only predicting countries 8, 10, 12

sum(predictions1$max_prob == predictions1$label) / nrow(predictions1) # 87.6% accuracy 

# variable importance
importance1 <- xgb.importance(colnames(train_matrix1), full_model1)
head(importance1)

first_201 <- importance1[1:20,]
xgb.plot.importance(first_201)


# variables used: 

# [1] "firstbook_y.1"                         "age_clean"                             "firstbook_snspring"                   
# [4] "affiliate_channelother"                "signup_flow"                           "gender_cleanFEMALE"                   
# [7] "gender_cleanMALE"                      "lag_acb_bin.1..365."                   "firstbook_y2014"                      
# [10] "age_bucket.1"                          "first_device_typeMac.Desktop"          "min_secs"                             
# [13] "median_secs"                           "lag_bts_bin.1..1369."                  "affiliate_providergoogle"             
# [16] "first_device_typeWindows.Desktop"      "sum_secs"                              "mean_secs"                            
# [19] "first_browserChrome"                   "std_secs"                              "signup_methodbasic"                   
# [22] "affiliate_channelsem.non.brand"        "firstbook_wkdSunday"                   "first_browserSafari"                  
# [25] "first_affiliate_trackedomg"            "firstbook_m05"                         "firstbook_m12"                        
# [28] "affiliate_providercraigslist"          "first_affiliate_trackeduntracked"      "acct_created_y2014"                   
# [31] "gender_cleanunknown"                   "firstbook_wkdMonday"                   "languagefr"                           
# [34] "affiliate_channeldirect"               "first_affiliate_trackedlinked"         "acct_created_snspring"                
# [37] "signup_appWeb"                         "acct_created_y2013"                    "firstbook_y2011"                      
# [40] "acct_created_wkdSunday"                "age_bucket20.24"                       "firstbook_snsummer"                   
# [43] "firstbook_wkdTuesday"                  "firstbook_y2012"                       "firstbook_wkdFriday"                  
# [46] "acct_created_wkdFriday"                "first_device_typeiPad"                 "firstbook_snfall"                     
# [49] "acct_created_wkdMonday"                "age_bucket25.29"                       "acct_created_wkdWednesday"            
# [52] "firstactive_d28"                       "age_bucket50.54"                       "max_secs"                             
# [55] "acct_created_snwinter"                 "acct_created_wkdSaturday"              "first_browserFirefox"                 
# [58] "d_num_Mac.Desktop"                     "acct_created_wkdThursday"              "firstbook_snwinter"                   
# [61] "firstactive_d16"                       "affiliate_channelcontent"              "firstbook_m07"                        
# [64] "firstactive_d25"                       "acct_created_wkdTuesday"               "acct_created_snsummer"                
# [67] "at_num_data"                           "firstbook_y2010"                       "firstbook_wkdWednesday"               
# [70] "firstbook_m08"                         "firstbook_wkdThursday"                 "acct_created_y2012"                   
# [73] "acct_created_m05"                      "first_browserMobile.Safari"            "firstactive_d13"                      
# [76] "first_affiliate_trackedproduct"        "firstbook_wkdSaturday"                 "first_browserIE"                      
# [79] "num_show"                              "firstactive_d06"                       "firstactive_d22"                      
# [82] "languageit"                            "acct_created_snfall"                   "signup_appMoweb"                      
# [85] "affiliate_channelsem.brand"            "ad_num_view_search_results"            "affiliate_providerfacebook"           
# [88] "firstactive_d23"                       "firstbook_m02"                         "acct_created_m11"                     
# [91] "acct_created_m08"                      "acct_created_m06"                      "firstbook_m04"                        
# [94] "firstactive_d09"                       "firstbook_y2013"                       "age_bucket30.34"                      
# [97] "acct_created_m03"                      "firstbook_m06"                         "age_bucket40.44"                      
# [100] "firstactive_d30"                       "firstactive_d10"                       "acct_created_m04"                     
# [103] "affiliate_channelremarketing"          "firstactive_d07"                       "firstactive_d18"                      
# [106] "ad_num_p3"                             "affiliate_channelseo"                  "firstactive_d02"                      
# [109] "firstactive_d14"                       "firstactive_d19"                       "acct_created_m09"                     
# [112] "firstactive_d17"                       "languageen"                            "firstbook_m03"                        
# [115] "firstactive_d05"                       "firstactive_y2014"                     "firstactive_d11"                      
# [118] "at_num_view"                           "age_bucket55.59"                       "languagees"                           
# [121] "acct_created_m07"                      "at_num_click"                          "firstbook_y2015"                      
# [124] "num_ajax_refresh_subtotal"             "firstactive_d08"                       "firstactive_d24"                      
# [127] "first_affiliate_trackedtracked.other"  "firstactive_d04"                       "acct_created_y2011"                   
# [130] "firstactive_d15"                       "d_num_Windows.Desktop"                 "firstactive_d21"                      
# [133] "firstbook_m01"                         "firstbook_m09"                         "first_device_typeOther.Unknown"       
# [136] "acct_created_y2010"                    "age_bucket35.39"                       "languagede"                           
# [139] "firstactive_d26"                       "ad_num_"                               "num_calendar_tab_inner2"              
# [142] "affiliate_providerother"               "num_similar_listings"                  "firstbook_m10"                        
# [145] "acct_created_m02"                      "acct_created_m01"                      "affiliate_providerbing"               
# [148] "languageko"                            "firstactive_d01"                       "firstactive_d03"                      
# [151] "firstactive_d31"                       "num_update"                            "affiliate_providervast"               
# [154] "firstactive_d29"                       "num_index"                             "ad_num_.unknown."                     
# [157] "acct_created_m10"                      "age_bucket45.49"                       "num_search_results"                   
# [160] "num_lookup"                            "first_device_typeAndroid.Tablet"       "firstactive_m04"
# [163] "num_edit"                              "first_browser.unknown."                "num_dashboard"                        
# [166] "languagept"                            "firstbook_m11"                         "acct_created_m12"                     
# [169] "num_create"                            "signup_methodfacebook"                 "firstactive_m05"                      
# [172] "at_num_submit"                         "firstactive_y2012"                     "affiliate_providerfacebook.open.graph"
# [175] "firstactive_wkdFriday"                 "firstactive_d20"                       "signup_appAndroid"                    
# [178] "signup_appiOS"                         "first_affiliate_tracked"               "first_device_typeDesktop..Other."     
# [181] "first_browserChrome.Mobile"            "ad_num_update_user_profile"            "firstactive_d27"                      
# [184] "affiliate_channelapi"                  "languagenl"                            "age_bucket70.74"                      
# [187] "first_device_typeiPhone"               "affiliate_providerdirect"              "firstactive_y2011"                    
# [190] "affiliate_providerpadmapper"           "firstactive_wkdSaturday"               "languagezh"                           
# [193] "num_populate_help_dropdown"            "age_bucket60.64"                       "firstactive_wkdSunday"                
# [196] "age_bucket65.69"                       "firstactive_y2010"                     "firstactive_m07"                      
# [199] "firstactive_d12"                       "num_requested"                        
# 
