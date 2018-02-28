
# ========================================================================================================
# Feature engineering 
# ========================================================================================================


# load data + packages
dir <- file.path(getwd(),"data")
# age_gender <- read.csv(file.path(dir, "age_gender_bkts.csv"))
# countries <- read.csv(file.path(dir, "countries.csv"))
sessions <- read.csv(file.path(dir, "sessionsSample.csv"))
sessions <- sessions[, -1] # removing extra X column 
train_users <- read.csv(file.path(dir, "train_users_2.csv"))

library(magrittr)
library(dplyr)
library(rebus)
library(caret)

# converting all factors to character
# countries <- dplyr::mutate_if(countries, is.factor, as.character)
sessions <- dplyr::mutate_if(sessions, is.factor, as.character)
train_users <- dplyr::mutate_if(train_users, is.factor, as.character)


# --------------------------------------------------------------------------------------------------------
# date account created
# --------------------------------------------------------------------------------------------------------


# converting to date class 
train_users$acct_created_date <- as.Date(train_users$date_account_created, format="%Y-%m-%d")

# pulling apart month + year of date created 
train_users$acct_created_y <- format(train_users$acct_created_date, "%Y")
train_users$acct_created_m <- format(train_users$acct_created_date, "%m")

# day of the week
train_users$acct_created_wkd <- weekdays(train_users$acct_created_date)

# season
train_users$acct_created_sn[as.numeric(train_users$acct_created_m) >= 3 & 
                              as.numeric(train_users$acct_created_m) <= 6] <- "spring"
train_users$acct_created_sn[as.numeric(train_users$acct_created_m) >= 7 & 
                              as.numeric(train_users$acct_created_m) <= 9] <- "summer"
train_users$acct_created_sn[as.numeric(train_users$acct_created_m) >= 10 & 
                              as.numeric(train_users$acct_created_m) <= 11] <- "fall"
train_users$acct_created_sn[as.numeric(train_users$acct_created_m) == 12 |  
                              as.numeric(train_users$acct_created_m) <= 2] <- "winter"


# --------------------------------------------------------------------------------------------------------
# timestamp first active 
# --------------------------------------------------------------------------------------------------------


# first active year 
train_users$firstactive_y <- stringr::str_sub(train_users$timestamp_first_active, start = 1, end = 4)

# first active month 
train_users$firstactive_m <- stringr::str_sub(train_users$timestamp_first_active, start = 5, end = 6)

# first active day 
train_users$firstactive_d <- stringr::str_sub(train_users$timestamp_first_active, start = 7, end = 8)

# first active full date 
train_users$firstactive_date <- as.Date(paste(train_users$firstactive_y, "-", 
                                              train_users$firstactive_m, "-", 
                                              train_users$firstactive_d, sep = ""), 
                                        format = "%Y-%m-%d")

# first active weekday 
train_users$firstactive_wkd <- weekdays(train_users$firstactive_date)


# --------------------------------------------------------------------------------------------------------
# date first booking
# --------------------------------------------------------------------------------------------------------


# converting to class = date
train_users$firstbook_date <- as.Date(train_users$date_first_booking, format="%Y-%m-%d")

# pulling apart month + year 
train_users$firstbook_y <- format(train_users$firstbook_date, "%Y")
train_users$firstbook_m <- format(train_users$firstbook_date, "%m")

# day of the week
train_users$firstbook_wkd <- weekdays(train_users$firstbook_date)

# season 
train_users$firstbook_sn[as.numeric(train_users$firstbook_m) >= 3 & 
                           as.numeric(train_users$firstbook_m) <= 6] <- "spring"
train_users$firstbook_sn[as.numeric(train_users$firstbook_m) >= 7 & 
                           as.numeric(train_users$firstbook_m) <= 9] <- "summer"
train_users$firstbook_sn[as.numeric(train_users$firstbook_m) >= 10 & 
                           as.numeric(train_users$firstbook_m) <= 11] <- "fall"
train_users$firstbook_sn[as.numeric(train_users$firstbook_m) == 12 |  
                           as.numeric(train_users$firstbook_m) <= 2] <- "winter"


# --------------------------------------------------------------------------------------------------------
# lag of date variables 
# --------------------------------------------------------------------------------------------------------


# days between date account created and date of first booking 
train_users$lag_acb <- train_users$firstbook_date - train_users$acct_created_date

# divide lag_acb into 4 bins: (0, [1,365], [-346, 0], NA)
train_users$lag_acb_bin[train_users$lag_acb == 0] <- "0"
train_users$lag_acb_bin[train_users$lag_acb > 0] <- "[1, 365]"
train_users$lag_acb_bin[train_users$lag_acb < 0] <- "[-349, 0)"
train_users$lag_acb_bin[is.na(train_users$lag_acb)] <- "NA"


# days between date first booking and time stamp first active
train_users$lag_bts <- train_users$firstbook_date - train_users$firstactive_date

# divide lag_bts into 3 bins: (0, [1, 1369], NA)
train_users$lag_bts_bin[train_users$lag_bts == 0] <- "0"
train_users$lag_bts_bin[train_users$lag_bts > 0] <- "[1, 1369]"
train_users$lag_bts_bin[is.na(train_users$lag_bts)] <- "NA"


# --------------------------------------------------------------------------------------------------------
# age 
# --------------------------------------------------------------------------------------------------------


# for users with ages in the 1900s, ageClean = year account created - age 
train_users$age_clean <- train_users$age
ages <- which(train_users$age > 1000 & train_users$age < 2000)
for (i in ages) {
  train_users$age_clean[i] <- as.numeric(train_users$acct_created_y[i]) - train_users$age[i] 
}

# setting all users with ages > 2000 or < 15 as -1
train_users$age_clean[train_users$age < 15 | train_users$age > 110 | is.na(train_users$age)] <- -1

# age ranges 
# 18 different age ranges (same as ranges in age_gender df)
train_users$age_bucket <- cut(train_users$age_clean,
                              breaks = c(0, seq(4, 104, by = 5)),
                              right = TRUE)


# --------------------------------------------------------------------------------------------------------
# gender
# --------------------------------------------------------------------------------------------------------


# replacing -unknown- and OTHER with empty string
train_users$gender_clean <- stringr::str_replace(train_users$gender, 
                                                 pattern = rebus::or("-unknown-", "OTHER"), 
                                                 replacement = "unknown")


# --------------------------------------------------------------------------------------------------------
# merge countries with train_users 
# --------------------------------------------------------------------------------------------------------


# train_users <- train_users %>% dplyr::full_join(countries, by = "country_destination")


# --------------------------------------------------------------------------------------------------------
# merge age_gender with train_users
# --------------------------------------------------------------------------------------------------------


# convert gender to lower case to resemble gender in age_gender df
# train_users$gender_clean <- stringr::str_to_lower(train_users$gender_clean)

# reformat age_bucket to resemble age_bucket in age_gender
train_users$age_bucket <- as.character(plyr::mapvalues(train_users$age_bucket,
                                                       from = levels(train_users$age_bucket),
                                                       to = c("0-4", "5-9", "10-14", "15-19",
                                                              "20-24", "25-29", "30-34", "35-39",
                                                              "40-44", "45-49", "50-54", "55-59",
                                                              "60-64", "65-69", "70-74", "75-79",
                                                              "80-84", "85-89", "90-94", "95-99",
                                                              "100+")))

# # for df merging: 
# age_gender <- dplyr::mutate_if(age_gender, is.factor, as.character)
# age_gender$gender_clean <- age_gender$gender
# 
# # merging age_gender with train_users according to country
# train_users <- train_users %>% left_join(age_gender[, c(1, 2, 4, 6)], 
#                                          by = c("country_destination", "age_bucket", "gender_clean"))


# --------------------------------------------------------------------------------------------------------
# sessions data 
# --------------------------------------------------------------------------------------------------------


# action - count occurences for each user 

# intialize empty data frame - append counts for each user 
counts <- data.frame(matrix(NA, nrow = 1, 
                            ncol = length(unique(sessions$action)) + 1))
colnames(counts) <- c("id", paste("num", unique(sessions$action), sep = "_"))



# function to count occurences of each type of action for each user 
get_action_counts <- function(user) {
  
  df <- sessions %>% filter(user_id == user) %>% count(action) # subsetting rows
  
  if (nrow(df) <= 1) { 
    
    df_t <- as.data.frame(data.table::transpose(df))  
    df_t$REMOVE_LATER <- NA # adding extra column (to retain df class)
    colnames(df_t) <- c(paste("num", df$action, sep = "_"), "REMOVE_LATER") # adding column names 
    df_t <- df_t[-1, ] # removing unecessary first row 
    
  } else {
    
    df_t <- as.data.frame(data.table::transpose(df))[-1, ] 
    colnames(df_t) <- paste("num", df$action, sep = "_") 
    
  }
  
  df_t <- dplyr::mutate_if(df_t, is.character, as.numeric)  
  df_t$id <- user # adding user id column
  
  return(df_t)
}



# initialize empty list
nums <- list()
unique_users <- unique(sessions$user_id) # iterate over each unique user and append to list 

for(i in 1:length(unique_users)) {
  nums[[i]] <- get_action_counts(unique_users[i])
}

counts <- counts %>% dplyr::bind_rows(nums) # bind data frames in nums to counts df
counts <- counts[-1, ] # remove first row of NAs
counts <- counts[, -195] # remove REMOVE_LATER column 

# merging counts with train_users
train_users <- train_users %>% full_join(counts, by = "id")


# --------------------------------------------------------------------------------------------------------


# action type counts

# empty data frame
type_counts <- data.frame(matrix(NA, nrow = 1, 
                                 ncol = length(unique(sessions$action_type)) + 1))
colnames(type_counts) <- c("id", paste("num", unique(sessions$action_type), sep = "_"))


# function to get counts for each var 
get_counts <- function(user, var) {
  
  df <- sessions %>% filter(user_id == user) %>% count_(var) # subsetting rows to each user
  
  if (nrow(df) <= 1) {
    
    df_t <- as.data.frame(data.table::transpose(df))
    df_t$REMOVE_LATER <- NA # adding extra column to retain df class
    colnames(df_t) <- c(paste("num", df[[var]], sep = "_"), "REMOVE_LATER") # adding column names
    df_t <- df_t[-1, ] # removing unecessary first row
    
  } else {
    
    df_t <- as.data.frame(data.table::transpose(df))[-1, ]
    colnames(df_t) <- paste("num", df[[var]], sep = "_")
  
  }
  
  df_t <- dplyr::mutate_if(df_t, is.character, as.numeric)
  df_t$id <- user # adding user id column
  
  return(df_t)
}


# initialize empty list, append counts to list 
num_types <- list()

for(i in 1:length(unique_users)) {
  num_types[[i]] <- get_counts(unique_users[i], "action_type")
}


# function to turn list into df + remove certain columns 
merge_counts <- function(df, count_list, var_name) {
  
  df <- df %>% dplyr::bind_rows(count_list)
  #df <- df[-1, ] # removing first row of NAs
  df <- df[, -which(colnames(df) == "REMOVE_LATER")] # removing REMOVE_LATER
  colnames(df) <- c("id", paste(var_name, colnames(df)[-1], sep = "_")) # adding identifier to column names
  
  return(df)

} 


type_counts <- merge_counts(type_counts, num_types, var_name = "at")
type_counts <- type_counts[-1, ]

# merging with train_users
train_users <- train_users %>% full_join(type_counts, by = "id")


# --------------------------------------------------------------------------------------------------------


# action detail

# empty data frame
detail_counts <- data.frame(matrix(NA, nrow = 1, 
                                   ncol = length(unique(sessions$action_detail)) + 1))
colnames(detail_counts) <- c("id", paste("num", unique(sessions$action_detail), sep = "_"))


num_details <- list()
for(i in 1:length(unique_users)) {
  num_details[[i]] <- get_counts(unique_users[i], "action_detail")
}

detail_counts <- merge_counts(detail_counts, num_details, var_name = "ad")
detail_counts <- detail_counts[-1, ]

# merging with train_users
train_users <- train_users %>% full_join(detail_counts, by = "id")


# --------------------------------------------------------------------------------------------------------


# device counts

device_counts <- data.frame(matrix(NA, nrow = 1, 
                                   ncol = length(unique(sessions$device_type)) + 1))
colnames(device_counts) <- c("id", paste("num", unique(sessions$device_type), sep = "_"))
# changing spaces in column names to "_"
colnames(device_counts) <- stringr::str_replace(colnames(device_counts), pattern = " ", replacement = "_") 

num_devices <- list()
for(i in 1:length(unique_users)) {
  num_devices[[i]] <- get_counts(unique_users[i], "device_type")
}

device_counts <- merge_counts(device_counts, num_devices, var_name = "d")
device_counts <- device_counts[-1,]

# merging with train_users
train_users <- train_users %>% full_join(device_counts, by = "id")

# changing NAs to 0
first <- which(colnames(train_users) == "num_index")
train_users[, first:ncol(train_users)][is.na(train_users[, first:ncol(train_users)])] <- 0


# --------------------------------------------------------------------------------------------------------
# seconds elapsed for each user
# --------------------------------------------------------------------------------------------------------


# compute summary statistics of secs_elapsed for each unique user 
secs_elapsed <- sessions %>% group_by(user_id) %>% summarise(sum_secs = sum(secs_elapsed, na.rm = TRUE),
                                                             mean_secs = mean(secs_elapsed, na.rm = TRUE), 
                                                             median_secs = median(secs_elapsed, na.rm = TRUE),
                                                             std_secs = sd(secs_elapsed, na.rm = TRUE), 
                                                             min_secs = min(secs_elapsed, na.rm = TRUE),
                                                             max_secs = max(secs_elapsed, na.rm = TRUE))

# removing row with no user_id
secs_elapsed <- secs_elapsed[-which(secs_elapsed$user_id == ""), ]

# rename user_id to id (to match train_users)
colnames(secs_elapsed)[1] <- "id"

# recode NAs to -1 
secs_elapsed[is.na(secs_elapsed)] <- -1

# merge with train_users 
train_users <- train_users %>% left_join(secs_elapsed, by = "id")


# save train_users as csv file
write.csv(train_users, file = "train_users.csv")


# --------------------------------------------------------------------------------------------------------
# one hot encoding
# --------------------------------------------------------------------------------------------------------

# encode all except: id, date_account_created, timestamp_first_active, date_first_booking, gender,
#                    age, country_destination, acct_created_date, firstactive_date, firstbook_date, 
#                    book, lat_destination, lng_destination, destination_km2, all sessions count variables, secs_elapsed variables

# removing row with NA for country_destination
train_users <- train_users[-which(is.na(train_users$country_destination)), ]

to_encode <- train_users %>% select(-c(starts_with("num_"), starts_with("d_"), starts_with("ad_"), starts_with("at_"), 
                                       id, date_account_created, timestamp_first_active, date_first_booking, gender, 
                                       age, country_destination, acct_created_date, firstactive_date, 
                                       firstbook_date, ends_with("secs"), lag_acb, lag_bts))

# converting all NAs to -1 (gbm doesn't work with NAs)
to_encode[is.na(to_encode)] <- -1 

train <- cbind(train_users$country_destination,
               data.frame(predict(dummyVars("~.", data = to_encode), newdata = to_encode)),
               train_users %>% select(starts_with("num_"), starts_with("d_"),
                                      starts_with("ad_"), starts_with("at_"), 
                                      ends_with("secs")))

colnames(train)[1] <- "country_destination"

# saving as csv 
write.csv(x = train, file = "train.csv")
x <- data.frame(predict(dummyVars("~.", data = train_users %>% select(age_bucket)), newdata = train_users %>% select(age_bucket)))

