# =================================================================================================================
# Random Forest
# =================================================================================================================

library(caret)
library(randomForest)
library(dplyr)
library(magrittr)

# load data
dir <- file.path(getwd(),"data")
train <- read.csv(file.path(dir, "train.csv"))

seed <- 100
set.seed(seed)

# removing first column 
train <- train[,-1]
# converting all column that are integers to numeric
train <- mutate_if(train, is.integer, as.numeric)

# =================================================================================================================

# setting up training and test data

# training data 
train_index <- caret::createDataPartition(y = train$country_destination, p = 0.70, list = FALSE)
train_data <- train[train_index[,1], ]

# test data 
test_data <- train[-train_index[,1], -1]
test_label <- train[-train_index[,1], 1]

# =================================================================================================================

# random forest model 

# with default parameters
# mtry: sqrt(number of predictors)
# default ntree is 500 - takes a long time, ntree = 50 still takes a long time 

rf_model <- randomForest(country_destination ~ ., data = train_data, ntree = 50, importance = TRUE, do.trace = 10)
rf_model # error rate: 12.46%, accuracy: 87.54% (out-of-bag, similar to CV)

# predictions on test data

pred <- predict(rf_model, newdata = test_data)
table(pred, test_label) # only predicting NDF, US, other, FR

sum(pred == test_label) / length(pred)
# 87.6% accuracy 

# =================================================================================================================

# feature importance

# Mean decrease accuracy: 
# based on hypothesis that if a feature is not important, then rearranging values of that feature will not harm accuracy 
# for each tree prediction error calculated on out-of-bag portion of the data is recorded, then the same is done after permuting
# each predictor variable - difference between the two are averaged over all trees, and normalized by standard dev of differences 

# Mean decrease gini
# total decrease in node impurities from splitting on the variable, averaged over all trees

feature_imp <- rf_model$importance
feature_imp <- feature_imp[order(-feature_imp[, ncol(feature_imp)]), ] # decreasing order 

# =================================================================================================================
# new random forest (on only most important features)
# =================================================================================================================

include <- rownames(feature_imp)[feature_imp[, ncol(feature_imp)] > 5] # features to include (with meandecreasegini > 5)

new_train <- train %>% select(country_destination, include)
# save as csv file for stacking 
# write.csv(new_train, file = "rf_train.csv")

# training data 
new_train_data <- new_train[train_index[,1], ]

# test data 
new_test_data <- new_train[-train_index[,1], -1]
new_test_label <- new_train[-train_index[,1], 1]

# refit random forest model 

new_rf_model <- randomForest(country_destination ~ ., data = new_train_data, ntree = 50, importance = TRUE, do.trace = 10)
new_rf_model # 12.57% error rate, 87.43% accuracy (out-of-bag, similar to CV)

# predict on test set 
new_pred <- predict(new_rf_model, newdata = new_test_data)
table(new_pred, new_test_label)

sum(new_pred == new_test_label) / length(new_pred) # 87.5% accuracy 
# only predicting: NDF, US, FR, other, AU, ES, IT 


rf_vars <- colnames(new_train)[-1] # for stacking (251 features)

# variables used: 

# [1] "firstbook_y.1"                         "age_clean"                             "affiliate_channelother"               
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


