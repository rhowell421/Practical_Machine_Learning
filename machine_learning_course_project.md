---
title: "Predicting Exercise Barbell Reps"
author: "Rhowell"
date: "2022-10-19"
output:
  html_document:
    keep_md: yes
    theme: "cosmo"
    toc: true
    toc_float: true
    toc_collapsed: true
  github_document: yes
---



## R Markdown

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Data Import


```r
# load libraries
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

# save URL as variable
train_url <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download data
download.file(url = train_url, destfile = "train_df.csv")
download.file(url = test_url, destfile = "test_df.csv")

#Reading the file
train_df <- read.csv(file = "train_df.csv")
test_df <- read.csv(file = "test_df.csv")

rbind(dim(train_df),
  dim(test_df))
```

```
##       [,1] [,2]
## [1,] 19622  160
## [2,]    20  160
```

## Cleaning data

We will download and split the training dataset into training and validation.


```r
# only keep numeric columns that are not blank
train_cleaned <- train_df %>% 
    mutate(classe = as.factor(classe)) %>%
    select(-"X") %>%
    discard(is.character) %>% 
    select_if(~ !any(is.na(.)))

test_cleaned <- test_df %>% 
    select(-c("X", "problem_id")) %>%
    discard(is.character) %>% 
    select_if(~ !any(is.na(.)))

# split training into training and validation sets
set.seed(45454)
in_train <- createDataPartition(train_cleaned$classe, p=0.70, list=F)
train_split <- train_cleaned[in_train, ]
validation_data <- train_cleaned[-in_train, ]
```

## k-Nearest Neighbors

Now we have the cleaned data sets, we will use the first model as k Nearest Neighbor using 5-fold cross-validation.


```r
# normalize data with min-max
min_max_norm <- function(x) {
     (x - min(x)) / (max(x) - min(x))
}
 
train_norm <- as.data.frame(lapply(train_split[,-56], min_max_norm)) %>%
   cbind(train_split[56])
validation_norm <- as.data.frame(lapply(validation_data[,-56], min_max_norm)) %>%
   cbind(validation_data[,56])
 
# check for NAs
rbind(anyNA(train_norm),
   anyNA(validation_norm))
```

```
##       [,1]
## [1,] FALSE
## [2,] FALSE
```

```r
# train kNN
trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
set.seed(12121)
knn_fit <- train(classe ~., data = train_norm, method = "knn",
    trControl=trctrl,
    preProcess = c("center"),
    tuneLength = 10)
```

Accuracy is 0.9422966 with k optimized at 5.

Let's test on the validation set.


```r
predict_knn <- predict(knn_fit, validation_norm)
confusionMatrix(validation_data$classe, predict_knn)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1338  206   84   29   17
##          B   96  853   73   48   69
##          C   64  275  570   65   52
##          D   49  246   62  533   74
##          E   34  202   45   42  759
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6887          
##                  95% CI : (0.6767, 0.7005)
##     No Information Rate : 0.3028          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6059          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8463   0.4787  0.68345  0.74338   0.7817
## Specificity            0.9219   0.9303  0.90972  0.91660   0.9343
## Pos Pred Value         0.7993   0.7489  0.55556  0.55290   0.7015
## Neg Pred Value         0.9423   0.8043  0.94567  0.96261   0.9559
## Prevalence             0.2686   0.3028  0.14172  0.12184   0.1650
## Detection Rate         0.2274   0.1449  0.09686  0.09057   0.1290
## Detection Prevalence   0.2845   0.1935  0.17434  0.16381   0.1839
## Balanced Accuracy      0.8841   0.7045  0.79659  0.82999   0.8580
```

```r
accuracy_knn <- postResample(predict_knn, validation_data$classe)
knn_oose <- 1 - as.numeric(confusionMatrix(validation_data$classe, predict_knn)$overall[1])
```

Accuracy is 0.6887, and estimated out of sample error is 0.3113.

## Random Forest

After looking at kNN, we will use Random Forest, another classification algorithm, to see how it compares. We will use 5-fold cross validation.


```r
controlRf <- trainControl(method="cv", 5)
rf_fit <- train(classe ~ ., data=train_split, method="rf", trControl=controlRf, ntree=100)
rf_fit
```

```
## Random Forest 
## 
## 13737 samples
##    55 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10990, 10987, 10991, 10990 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9950495  0.9937378
##   28    0.9986894  0.9983423
##   55    0.9968699  0.9960409
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 28.
```

Accuracy is 0.9986894 with trees optimized at 28.



```r
predict_rf <- predict(rf_fit, validation_data)
confusionMatrix(validation_data$classe, predict_rf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    1 1138    0    0    0
##          C    0    0 1026    0    0
##          D    0    0    1  962    1
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9995          
##                  95% CI : (0.9985, 0.9999)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9994          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   1.0000   0.9990   1.0000   0.9991
## Specificity            1.0000   0.9998   1.0000   0.9996   1.0000
## Pos Pred Value         1.0000   0.9991   1.0000   0.9979   1.0000
## Neg Pred Value         0.9998   1.0000   0.9998   1.0000   0.9998
## Prevalence             0.2846   0.1934   0.1745   0.1635   0.1840
## Detection Rate         0.2845   0.1934   0.1743   0.1635   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9999   0.9995   0.9998   0.9995
```

```r
accuracy_rf <- postResample(predict_rf, validation_data$classe)
oose_rf <- 1 - as.numeric(confusionMatrix(validation_data$classe, predict_rf)$overall[1])
```

Accuracy is 0.9995, and estimated out of sample error is 5.097706\times 10^{-4}.

## Predicting the test set

To predict the results of the test, we will use the Random Forest model.


```r
predict_test <- predict(rf_fit, test_cleaned)
predict_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


