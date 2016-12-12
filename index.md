# Practical Machine Learning - Course Project
Kyle McEligot  


##Executive Summary

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

The goal of the project is to predict the manner in which they did the exercise.

### Data provided courtesy of:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz3vWGDyDo2

### Load Needed Libraries

```r
library(caret, quietly = TRUE)
library(rpart, quietly = TRUE)
suppressMessages(library(randomForest, quietly = TRUE, verbose = FALSE))
suppressMessages(library(survival, quietly = TRUE, verbose = FALSE))
suppressMessages(library(gbm, quietly = TRUE, verbose = FALSE))
library(plyr, quietly = TRUE)
```

## Data Loading

```r
plm.training <- read.csv("Data/pml-training.csv", header=T,
                         na.strings=c("", "NA", "NULL"))
plm.testing <- read.csv("Data/pml-testing.csv",header=T,
                         na.strings=c("", "NA", "NULL"))

dim(plm.training)
```

```
## [1] 19622   160
```

```r
dim(plm.testing)
```

```
## [1]  20 160
```

## Cleaning Data

Exploratory Data Analysis showed that:  

* many of the columns had many NA values
    + note: "" and "NULL" were converted to NA with read.csv
* the first 7 columns contain variables that are not relevant to the dependent variable
* many of the columns had many NA values (with non being completely NA)
* some columns were highly correlated


```r
# Remove the first 7 columns.
training.set <- plm.training[, 
                -which(names(plm.training) %in% 
                 c('X',
                  'user_name',
                  'raw_timestamp_part_1',
                  'raw_timestamp_part_2', 
                  'cvtd_timestamp',                         
                  'new_window', 
                  'num_window'))]

dim(training.set)
```

```
## [1] 19622   153
```

```r
# Remove columns with many NA values
datavariables <- 
  apply(!is.na(training.set),2,sum) >= dim(training.set)[1]
training.set <- training.set[, datavariables]

dim(training.set)
```

```
## [1] 19622    53
```

```r
# Remove columns that are highly correlated

correlation.info <- 
  cor(na.omit(training.set[
                sapply(training.set, is.numeric)]))
training.set <- training.set[,
                             -c(findCorrelation(correlation.info, 
                                                cutoff = .90))]
dim(training.set)
```

```
## [1] 19622    46
```


## Models

Three different models will be trained:

1. Decision Trees
2. Random Forest
3. Boosting

The best performing model will be selected. 

For this approach, the test dataset will be split into two: train and predict. 



```r
set.seed(1234)
 
inTrain <- createDataPartition(y=training.set$classe, 
                               p = 0.7, list=FALSE)

train <- training.set[inTrain,]
probe <- training.set[-inTrain,]

dim(train)
```

```
## [1] 13737    46
```

```r
dim(probe)
```

```
## [1] 5885   46
```
Cross validation is set for the models through trainControl

```r
# Set up the trainControl for all models 
#   with method cv (cross validation) at 3
trControl = trainControl(method="cv", number=3)
```
### Decision Tree


```r
# Start the clock!
ptm <- proc.time()

decision.tree.model <- train(classe ~ .,
                             data=train,
                             trControl=trControl,
                             method='rpart')

# Stop the clock
proc.time() - ptm
```

```
##    user  system elapsed 
##    4.16    0.15    4.36
```

```r
decision.tree.predict <- predict(decision.tree.model, 
                                 newdata = probe)
confusionMatrix.decision.tree <- 
  confusionMatrix(decision.tree.predict, probe$classe)
```

### Random Forest


```r
# Start the clock!
ptm <- proc.time()

random.forest.model <- train(classe ~ .,
                             data=train,
                             trControl=trControl,
                             method='rf')

# Stop the clock
proc.time() - ptm
```

```
##    user  system elapsed 
##  258.35    2.16  262.33
```

```r
random.forest.predict <- predict(random.forest.model, 
                                 newdata = probe)
confusionMatrix.random.forest <- 
  confusionMatrix(random.forest.predict, probe$classe)
```


### Boosting


```r
# Start the clock!
ptm <- proc.time()

boosting.model <- train(classe ~ .,
                        data=train,
                        trControl=trControl,
                        verbose=FALSE,
                        method='gbm')

# Stop the clock
proc.time() - ptm
```

```
##    user  system elapsed 
##  127.06    0.32  127.82
```

```r
boosting.predict <- predict(boosting.model, 
                                 newdata = probe)
confusionMatrix.boosting <- 
  confusionMatrix(boosting.predict, probe$classe)
```


## Evaluating the Models

To evaluate the models, look at
 - accuracy (from confusion matrix)  
 - out of sample error for model types  


```r
data.frame(
  Model.Type = c('Decision Tree', 'Random Forest', 'Boosting'),
  Accuracy = rbind(confusionMatrix.decision.tree$overall['Accuracy'],
                   confusionMatrix.random.forest$overall['Accuracy'],
                   confusionMatrix.boosting$overall['Accuracy']))
```

```
##      Model.Type  Accuracy
## 1 Decision Tree 0.4999150
## 2 Random Forest 0.9955820
## 3      Boosting 0.9587086
```

We can see that random forest is most accurate at 
0.995582


```r
confusionMatrix.decision.tree$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1038  210   30   47   21
##          B  161  608  330  320  411
##          C  322  259  661  227  243
##          D  151   61    5  300   72
##          E    2    1    0   70  335
```

```r
confusionMatrix.random.forest$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    8    0    0    0
##          B    0 1128    4    1    0
##          C    0    3 1020    5    1
##          D    0    0    2  957    1
##          E    0    0    0    1 1080
```

```r
confusionMatrix.boosting$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1650   46    0    0    0
##          B   13 1058   34    5   13
##          C    6   29  974   26   16
##          D    3    2   14  920   13
##          E    2    4    4   13 1040
```

## Prediction on test data - using best performing model

The best performing model of the three was Random Forest. It will be used
on the testing data (pml-testing.csv) to predict a class for each of the 20 test cases.


```r
rf.test.prediction <- predict(random.forest.model,
                        newdata = plm.testing)
rf.test.prediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion

Of the three models tried, the random forest performed best, with very good accuracy.  That model was used to predict using the test data from the website.

