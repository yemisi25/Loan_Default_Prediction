library(odbc)
library(DBI)
library(dplyr)
library(tidyverse) 
library(tidyr)
library(hash)
library(glmnet)
library(Metrics)
library(MetricsWeighted)
library(caret)
library(pROC)
library(vtreat)
library(randomForest)
library(ROSE)
library(DMwR2)


# connect to the database using odbc
con <- dbConnect(odbc::odbc(),
                 Driver = "SQL Server",
                 Server = "OLUWAYEMISI\\SQLEXPRESS01", 
                 Database = "LoanDefault_",
                 Port = 1433)


# create a train and test objects for the train and test table in database
train_tbl <- tbl(con, 'Data_Train')
test_tbl <- tbl(con, 'Data_Test')

# collect all data from the train and test table objects
train <- collect(train_tbl)
test <- collect(test_tbl)

# CHECK FOR MISSING VALUES IN THE DATASET
colSums(is.na(train)) # there is no missing values in any of the column
colSums(is.na(test)) # there is no missing values in any of the column

# check for duplicate in train and test data: no duplicate data
sum(duplicated(train)) # no duplicate
sum(duplicated(test)) # no duplicate

# summarize the data
View(summary(train))
View(summary(test))

# obtain the numerical column and non numerical column: train data
numerical_train <- train %>% select_if(is.numeric)
non_numerical_train <- train %>% select_if(negate(is.numeric))

# size of the train numerical and non-numerical data 
dim(numerical_train) # rows:columns <> 87500:18
dim(non_numerical_train) # rows:columns <> 87500:12

# view the numerical train data
View(numerical_train)

# the number of TRUE and FALSE values in train data
paste("% of FALSE values in target variables:", 
      as.character(round((sum(train$Loan_No_Loan == FALSE) / nrow(train) * 100), 2))) # 81.13% 
paste("% of TRUE values in target variables:", 
      as.character(round((sum(train$Loan_No_Loan == TRUE) / nrow(train) * 100), 2))) # 18.87% 

################## Handle missing values in the train data set #######################################
miss_num <- as.data.frame(as.matrix(colSums(is.na(numerical_train))))
miss_num <- rename(miss_num, no_missing_values = V1)
miss_num$variables <- rownames(miss_num)
rownames(miss_num) <- NULL
miss_num
miss_num_names <- miss_num[miss_num['no_missing_values'] > 0, ]$variables
miss_num_names

# find the statistical value for the missing values
summary(numerical_train[, miss_num_names])
# check for the distribution
hist(numerical_train$Yearly_Income) # median
hist(numerical_train$Debt_to_Income) # mean
hist(numerical_train$Postal_Code) # missing values should be dropped 
hist(numerical_train$Total_Unpaid_CL) # median
hist(numerical_train$Unpaid_Amount) # median

# dictionary for variables with missing values and their aggregated function
dic <- hash(keys=miss_num_names[-3], values= c('median', 'mean', 'median', 'median'))

# fill the missing values of the numerical columns apart from Postal_Code

for (i in list(keys(dic))) {
  k = values(dic[i])[[1]] 
  if (k ==  'median') {
    train[i][is.na(train[i])] <- median(
      as.numeric(unlist(train[i])), na.rm=T)
  }
  else {
    train[i][is.na(train[i])] <- mean(
      as.numeric(unlist(train[i])), na.rm=T)
  }
}

colSums(is.na(train))

# check for the missing values in the non-numerical column
colSums(is.na(non_numerical_train)) # Already_Defaulted has 26 missing values: i will drop it

# drop all observations in train data set based on missing values in:
# postal_code and Already_Defaulted (1389 and 26 respectively: 1415)

train <- (train[complete.cases(train),])
View(train)
# check for missing values again
colSums(is.na(train))

############################## FEATURE SELECTION ######################################

# convert Loan_No_Loan to integers
new_numeric <- train[, names(numerical_train)]
new_numeric['Loan_No_Loan'] <- train$Loan_No_Loan

new_numeric$Loan_No_Loan <- as.numeric(new_numeric$Loan_No_Loan)
new_numeric$Loan_No_Loan

# non numerical
names(non_numerical_train)

ordinal_var <- c("GGGrade", "Experience", "Already_Defaulted", "Duration")
sapply(lapply(train[, ordinal_var], unique), length)

# encode some of the categorical variable
unique(non_numerical_train['Experience'])
gggrade <- hash(c('I', 'II', 'III', 'IV', 'V', 'VI', 'VII'), c(1,2,3,4,5,6,7))
duration <- hash(c('3 years', '5 years'), c(1,2))
experience <- hash(c('<1yr', '1yrs', '2yrs', '3yrs', '4yrs', '5yrs', '6yrs', 
                     '7yrs', '8yrs', '9yrs', '>10yrs'), 
                   c(1,2,3,4,5,6,7,8,9,10,11))

# create the function for encoding the numerical columns
for (k in keys(gggrade)) {
  train['GGGrade'][train['GGGrade'] == k] <- as.character(values(gggrade[k])[[1]])
}
new_numeric['GGGrade'] <- as.numeric(train$GGGrade)
for (k in keys(duration)) {
  train['Duration'][train['Duration'] == k] <- as.character(values(duration[k])[[1]])
}
new_numeric['Duration'] <- as.numeric(train$Duration)
for (k in keys(experience)) {
  train['Experience'][train['Experience'] == k] <- as.character(values(experience[k])[[1]])
}
new_numeric['Experience'] <- as.numeric(train$Experience)


View(new_numeric)


########################### EVALUATE MODEL WITH TRAIN TEST SPLIT ################################
set.seed(1)
gp <- runif(nrow(new_numeric))
# split the data into 20% test and 80% train
# train 80%
train.subset <- new_numeric[gp<=0.8, ][-1]

# check the level of classes in the target variable -> 55849 : 12992
train.subset %>% group_by(Loan_No_Loan) %>% summarise(category_size=length(Loan_No_Loan))

# test 20% split data
test.subset <- new_numeric[gp>0.8, ]
# test label
y_test <- new_numeric[gp>0.8, ]$Loan_No_Loan
length(y_test)
y_test

dim(train.subset)
dim(test.subset)

################################### MODEL SELECTION #######################################3
################################ STEP WISE MODEL ################################
# create a model with no predictor
null_model <-  glm(Loan_No_Loan ~ 1, data=train.subset, family='binomial')

# create a new model with all the numerical predictors available
full_model <-  glm(Loan_No_Loan ~., data=train.subset, family='binomial')

# use forward stepwise
step_model <- step(null_model, scope=list(lower=null_model, upper=full_model), direction='forward')

summary(step_model)
important_features <- names(coef(step_model))[-1]
important_features

################################3 CREATE NEW TRAIN WITH THE SELECTED FEATURES ######################
new_df <- new_numeric[, important_features]
new_df['Loan_No_Loan'] <- as.numeric(new_numeric$Loan_No_Loan)
names(new_df)

# split new_df
set.seed(1)
gp <- runif(nrow(new_df))
# split the data into 20% test and 80% train
train.subset1 <- new_df[gp<=0.8, ]

# using rose technique
train.subset.rose1 <- ROSE(Loan_No_Loan~., data=train.subset1)$data

test.subset1 <- new_df[gp>0.8, ]
# test label
y_test1 <- new_df[gp>0.8, ]$Loan_No_Loan


# RUN LOGISTIC REGRESSION MODEL
log.model <- glm(Loan_No_Loan ~., data=train.subset.rose1, family='binomial')
# predict on the test data
log.model.prob <- predict(log.model, test.subset1, type='response')
log.model.prob[1:10]
length(log.model.prob)
# set every value greater than zero to 1 else 0
log.model.pred <- ifelse(log.model.prob > 0.50, 1, 0)
log.model.pred[1:10]
# actual value
y_test[1:10]
length(y_test)
length(log.model.pred)
n_distinct(log.model.pred)
n_distinct(y_test)

# using confusion matrix to evaluate the model
confussion_mat <- confusionMatrix(data = factor(log.model.pred), reference = factor(y_test))

confussion_mat

################################# TRAIN ON WHOLDE TRAINING SET ##################################
# get the categorical column with less  than 5 categories
new_newer_train <- new_numeric[, important_features]
names(new_newer_train)
new_newer_train$Loan_No_Loan <- train$Loan_No_Loan
# create a factor for the target variable
new_newer_train$Loan_No_Loan <- factor(new_newer_train$Loan_No_Loan,
                                       levels = c('FALSE', 'TRUE'),
                                       labels =  c(0,1))

View(new_newer_train)

# using rose technique on whole training dataset
train.rose1 <- ROSE(Loan_No_Loan~., data=new_newer_train)$data


# fit the logistic regression on the whole training data set
log.model.final <- glm(Loan_No_Loan ~., data=train.rose1, family='binomial')



#clean the test data set
new_test <- test[, important_features]
colSums(is.na(new_test))
summary(train$Unpaid_Amount) # use median
summary(train$Yearly_Income) # use median

# the most ocurring postal code
new_test %>% group_by(Postal_Code) %>% summarise(count_=length(Postal_Code)) # highest: 1000

# fill missing values
new_test$Unpaid_Amount[is.na(new_test$Unpaid_Amount)] <- median(train$Unpaid_Amount, na.rm=T)
new_test$Yearly_Income[is.na(new_test$Yearly_Income)] <- median(train$Yearly_Income, na.rm=T)
new_test$Postal_Code[is.na(new_test$Postal_Code)] <- 1000

View(new_test)

# encode the duration column
for (k in keys(duration)) {
  new_test['Duration'][new_test['Duration'] == k] <- as.character(values(duration[k])[[1]])
}
new_test['Duration'] <- as.numeric(new_test$Duration)

# predict on the test whole test data using the model that was fitted with whole train
log.model.prob <- predict(log.model.final, new_test, type='response')
# actual true or false value
log.model.pred <- ifelse(log.model.prob > 0.50, 1, 0)

# create a new data frame for the predicted test data
pred_test_table <- data.frame(S.No=c(1:nrow(test)), Id=test$ID, prediction=log.model.pred)
unique(pred_test_table$prediction)
View(pred_test_table)
# add Loan_No_Loan column
# pred_test_table$Loan_No_Loan <- ifelse(pred_test_table$prediction == 0, FALSE, TRUE )

# save the predicted table as a csv file
write.csv(pred_test_table, "FinalPredictedLoanDefault.csv", row.names=FALSE)

####################################################################################################
pred_test_table %>% group_by(prediction) %>% summarise(count_=length(prediction))
train %>% group_by(Loan_No_Loan) %>% summarise(count_=length(Loan_No_Loan))

predict(log.model.final, new_numeric, type='response')


# EVALUATE THE TRAINING DATASET
baba <- new_numeric %>% group_by(Loan_No_Loan) %>% summarise(count_=length(Loan_No_Loan))
baba
baba_pred <- predict(log.model.final, new_numeric[, important_features], type='response')
baba_pred2 <- ifelse(baba_pred > 0.50, 1, 0)
baba_pred2

confussion_mat2 <- confusionMatrix(data = factor(baba_pred2), reference = factor(as.numeric(train$Loan_No_Loan)))
confussion_mat2


