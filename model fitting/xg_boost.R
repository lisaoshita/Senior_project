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
                   "num_class" = n_classes,
                   eta = 0.3, 
                   gamma = 0, 
                   max_depth = 6, 
                   min_child_weight = 1, 
                   subsample = 0.8, 
                   colsample_bytree = 0.9)
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

# Predict hold-out test set
heldout_test_pred <- predict(full_model, newdata = test_matrix)
predictions <- matrix(heldout_test_pred, 
                      nrow = n_classes, 
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

#  "firstbook_y.1"                         "age_clean"                             "affiliate_channelother"               
# [4] "firstbook_snspring"                    "firstbook_y2014"                       "signup_appWeb"                        
# [7] "gender_cleanMALE"                      "gender_cleanFEMALE"                    "first_device_typeWindows.Desktop"     
# [10] "age_bucket25.29"                       "firstbook_m12"                         "affiliate_channelsem.non.brand"       
# [13] "age_bucket20.24"                       "signup_flow"                           "firstbook_snwinter"                   
# [16] "languagefr"                            "first_browserSafari"                   "first_device_typeMac.Desktop"         
# [19] "firstbook_snfall"                      "firstbook_y2011"                       "first_browserChrome"                  
# [22] "firstbook_y2012"                       "firstbook_wkdSunday"                   "first_device_typeiPad"                
# [25] "affiliate_channelsem.brand"            "first_affiliate_trackedlinked"         "affiliate_providercraigslist"         
# [28] "firstbook_wkdMonday"                   "num_ajax_refresh_subtotal"             "signup_methodbasic"                   
# [31] "firstbook_m05"                         "d_num_Mac.Desktop"                     "acct_created_y2012"                   
# [34] "firstbook_y2013"                       "acct_created_snwinter"                 "affiliate_channelcontent"             
# [37] "languagede"                            "firstbook_wkdFriday"                   "firstbook_snsummer"                   
# [40] "acct_created_y2014"                    "languageit"                            "acct_created_wkdSunday"               
# [43] "ad_num_p3"                             "acct_created_snfall"                   "first_affiliate_trackedomg"           
# [46] "firstbook_m03"                         "age_bucket50.54"                       "firstbook_m02"                        
# [49] "firstbook_y2010"                       "firstactive_d14"                       "firstbook_wkdTuesday"                 
# [52] "languagees"                            "firstbook_m11"                         "first_affiliate_trackeduntracked"     
# [55] "age_bucket30.34"                       "first_browserIE"                       "first_affiliate_trackedproduct"       
# [58] "affiliate_providerother"               "firstbook_m01"                         "acct_created_m12"                     
# [61] "affiliate_channelseo"                  "languageko"                            "affiliate_providergoogle"             
# [64] "acct_created_y2013"                    "acct_created_m06"                      "languageen"                           
# [67] "acct_created_snsummer"                 "ad_num_view_search_results"            "firstactive_d23"                      
# [70] "acct_created_wkdWednesday"             "firstbook_m07"                         "firstactive_d10"                      
# [73] "firstactive_d20"                       "first_browserMobile.Safari"            "age_bucket35.39"                      
# [76] "acct_created_m07"                      "num_show"                              "signup_appMoweb"                      
# [79] "firstactive_d24"                       "acct_created_wkdMonday"                "firstactive_d12"                      
# [82] "acct_created_wkdTuesday"               "acct_created_m09"                      "num_lookup"                           
# [85] "gender_cleanunknown"                   "acct_created_m05"                      "firstactive_d16"                      
# [88] "first_device_typeiPhone"               "firstactive_d22"                       "acct_created_wkdSaturday"             
# [91] "firstbook_wkdSaturday"                 "firstbook_wkdThursday"                 "firstactive_d09"                      
# [94] "age_bucket.1"                          "acct_created_wkdFriday"                "firstbook_m09"                        
# [97] "firstbook_wkdWednesday"                "firstactive_d28"                       "firstactive_d13"                      
# [100] "acct_created_m03"                      "acct_created_m01"                      "firstbook_m10"                        
# [103] "affiliate_channeldirect"               "d_num_Windows.Desktop"                 "firstactive_wkdSunday"                
# [106] "acct_created_snspring"                 "affiliate_providervast"                "firstactive_d29"                      
# [109] "acct_created_wkdThursday"              "first_device_typeDesktop..Other."      "firstactive_d04"                      
# [112] "acct_created_m11"                      "acct_created_y2010"                    "firstactive_d25"                      
# [115] "ad_num_create_user"                    "firstactive_d17"                       "age_bucket15.19"                      
# [118] "num_index"                             "affiliate_channelremarketing"          "first_browserFirefox"                 
# [121] "acct_created_y2011"                    "acct_created_m04"                      "first_affiliate_tracked"              
# [124] "firstbook_m08"                         "firstactive_y2010"                     "ad_num_"                              
# [127] "firstactive_d03"                       "acct_created_m02"                      "firstactive_d07"                      
# [130] "firstactive_d02"                       "num_personalize"                       "languagept"                           
# [133] "age_bucket65.69"                       "firstactive_d15"                       "firstbook_y2015"                      
# [136] "firstactive_y2011"                     "firstactive_d18"                       "age_bucket55.59"                      
# [139] "first_browser.unknown."                "d_num_iPad.Tablet"                     "affiliate_providerbing"               
# [142] "ad_num_.unknown."                      "affiliate_providerfacebook"            "first_device_typeAndroid.Tablet"      
# [145] "firstbook_m04"                         "firstactive_d31"                       "firstbook_m06"                        
# [148] "firstactive_d08"                       "firstactive_d30"                       "acct_created_m08"                     
# [151] "num_search_results"                    "firstactive_d27"                       "firstactive_d26"                      
# [154] "languagenl"                            "age_bucket75.79"                       "acct_created_m10"                     
# [157] "firstactive_d05"                       "num_confirm_email"                     "num_calendar_tab_inner2"              
# [160] "affiliate_providerdirect"              "num_header_userpic"                    "firstactive_y2012"                    
# [163] "num_authenticate"                      "signup_appiOS"                         "firstactive_d21"                      
# [166] "firstactive_d01"                       "firstactive_m11"                       "num_faq_category"                     
# [169] "num_requested"                         "firstactive_wkdTuesday"                "num_show_personalize"                 
# [172] "age_bucket45.49"                       "age_bucket40.44"                       "affiliate_providerfacebook.open.graph"
# [175] "firstactive_y2013"                     "num_listings"                          "ad_num_user_wishlists"                
# [178] "firstactive_d11"                       "num_apply_reservation"                 "firstactive_m05"                      
# [181] "ad_num_update_user_profile"            "firstactive_d06"                       "firstactive_d19"                      
# [184] "firstactive_m06"                       "first_affiliate_trackedtracked.other"  "firstactive_wkdThursday"              
# [187] "num_profile_pic"                       "first_browserChrome.Mobile"            "firstactive_m12"                      
# [190] "num_open_graph_setting"                "first_device_typeOther.Unknown"        "affiliate_channelapi"                 
# [193] "firstactive_m02"                       "num_track_page_view"                   "num_dashboard"                        
# [196] "age_bucket60.64"                       "firstactive_m09"                       "num_reviews_new"                      
# [199] "num_verify"                            "num_active"                            "ad_num_post_checkout_action"          
# [202] "num_update"                            "num_reviews"                           "age_bucket85.89"                      
# [205] "first_device_typeAndroid.Phone"        "d_num_iPhone"                          "num_edit"                             
# [208] "num_similar_listings"                  "affiliate_providerpadmapper"           "firstactive_wkdMonday"                
# [211] "firstactive_wkdWednesday"              "d_num_.unknown."                       "first_browserChromium"                
# [214] "num_notifications"                     "num_"                                  "num_ask_question"                     
# [217] "num_hosting_social_proof"              "ad_num_update_listing"                 "num_my"                               
# [220] "signup_appAndroid"                     "d_num_Linux.Desktop"                   "num_collections"                      
# [223] "num_qt_with"                           "firstactive_wkdSaturday"               "num_this_hosting_reviews"             
# [226] "firstactive_wkdFriday"                 "num_unavailabilities"                  "affiliate_provideremail.marketing"    
# [229] "firstactive_m08"                       "num_qt2"                               "ad_num_update_listing_description"    
# [232] "ad_num_user_profile"                   "num_languages_multiselect"             "num_ajax_image_upload"                
# [235] "num_at_checkpoint"                     "num_settings"                          "firstactive_m01"                      
# [238] "num_complete"                          "languagesv"                            "num_available"                        
# [241] "num_12"                                "age_bucket70.74"                       "ad_num_p5"                            
# [244] "ad_num_signup"                        

