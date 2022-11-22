#DIABETES PREDICTION
#JOSE JAVIER MIÑANO RAMOS

#Installing dependencies
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")

#loading packages
library(ggplot2)
library(tidyverse)
library(caret)
library(readr)
library(data.table)
library(dplyr)
library(broom) 
library(rpart)
library(matrixStats)
library(gam)

#loading dataset
help("read.csv")
data = read.csv("diabetes_data.csv") #before running this line we have to make sure that we
                                  #have diabetes.csv in our working directory

  ### EXPLORING DATASET

dim(data)   #768x9
head(data)
summary(data)

#What we are trying to predict is the outcome column, 1 means diabetic, 0 means non diabetic.
#Now we are going to observe the distribution of ghe outcome column. Too many 1 or too many 0 would suggest a biased dataset
table(data$Outcome) #500 zeros, 268 ones

#group the data by diabetics or non diabetics, we can see how the mean insulin level is lower for non diabetics or mean pregnancies are higher for diabetics.
#valuable information:
diabetics <- data[data$Outcome == 1,]
  summary(diabetics)
non_diabetics <- data[data$Outcome == 0, ]
  summary(non_diabetics)

#now we are going to look for mean values of every variable:
sapply(diabetics, mean) 
sapply(non_diabetics, mean)

  ### PLOTS

#glucose density functions for diabetics and non diabetics
plot(density(diabetics$Glucose),ylim = c(0,0.022) ,col = "blue", main = "Glucose density functions")
lines(density(non_diabetics$Glucose), col = "red")
legend("topright", c("diabetics glucose density function", "non diabetics glucose density function"),
       col = c("blue", "red"), lty =1, cex= 0.8)

#insulin density function for diabetics and non diabetics
plot(density(diabetics$Insulin), ylim = c(0,0.011) ,col = "blue", main = "Insulin density function", xlab = "  ")
lines(density(non_diabetics$Insulin), col = "red")
legend("topright", c("diabetics insulin density function", "non diabetics insulin density function"),
       col = c("blue", "red"), lty =1, cex= 0.8)

#pregnancies histogram for diabetics and non diabetics
p1 <- hist(diabetics$Pregnancies)                     
p2 <- hist(non_diabetics$Pregnancies)                    
plot( p1, col=rgb(0,0,1,1/4), xlim=c(0,20), ylim=c(0,200), main = NULL, xlab = "Pregnancies")  # first histogram
plot( p2, col=rgb(1,0,0,1/4), xlim=c(0,10), add=T)  # second
legend("topright", c("diabetics pregnancy frequency", "non diabetics pregnancy frequency"),
       col = c("blue", "red"), lty =1, cex= 0.8)

  ### SEPARATING DATA AND LABELS, DATA STANDARDIZATION AND SPLITTING THE DATA IN TEST AND TRAIN DATA

  # SEPARATING DATA AND LABELS
X <- data[, 1:8] 
  dim(X) #768x8
Y <- data$Outcome
  length(Y) #768 elements

# calculate % of diabetics and non diabetics
mean(Y == 1) #35% of diabetics in the dataset

  # DATA STANDARDIZATION

# normalizing the data will improve our algorithms' precision

X_centered <- sweep(X, 2, colMeans(X), FUN ="-") #subtracting the mean in every column
X_scaled <- sweep(X_centered, 2, colSds(as.matrix(X)), FUN = "/") #dividing by the standard deviation
data_scaled <- cbind(X_scaled, Y)

#we will use the information of X scaled to train our algorithms

  # SPLIT THE DATA INTO TEST AND TRAIN DATA

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(Y, times = 1, p=0.25, list = FALSE) #We are going to train with the 75% of the data
test_set <- data_scaled[test_index,]
train_set <- data_scaled[-test_index, ]


mean(train_set$Y == 1)
mean(test_set$Y == 1) #diabetics proportion in both train and test set is similar (35% +- 0,7%)


  ### TRAINING MODELS

# We will start by guessing, as a baseline model
guess_model <- sample(c(0,1), length(test_index), replace = TRUE)
mean(guess_model == test_set$Y) #Accuracy of 55%

#logistic regression model (useful for classification)
fit_linear <- train(as.factor(Y)~., method = "glm", data = train_set)
linear_preds <- predict(fit_linear, test_set)

mean(linear_preds == test_set$Y)

#principal component analysis
pca_matrix <- prcomp(train_set[,-Y]) 
summary(pca_matrix) #pregnancies explain 30% of variance.

#prediction (lda model) only using the principal component (pregnancies)
fit_lda_preg <- train(as.factor(Y)~Pregnancies, method = "lda", data = train_set)
lda_preds_preg <- predict(fit_lda_preg, test_set)

mean(lda_preds_preg == test_set$Y)

#lda model using all the predictors:
?caret::train
fit_lda <- train(as.factor(Y)~., method = "lda", data = train_set)
lda_preds <- predict(fit_lda, test_set)

mean(lda_preds == test_set$Y)

#qda model:
fit_qda <- train(as.factor(Y)~., method = "qda", data = train_set)
qda_preds <- predict(fit_qda, test_set)

mean(qda_preds == test_set$Y)

          #qda model is less precise than lda model

#knn model
train_knn <- train(as.factor(Y) ~ ., 
                   method = "knn",
                   data = train_set,
                   tuneGrid = data.frame(k = seq(3, 51, 2)))
train_knn$bestTune
ggplot(train_knn)
knn_pred <- predict(train_knn, test_set)
mean(knn_pred == test_set$Y)


#Creating a ensemble of models
ensemble <- cbind(glm = linear_preds == 1, lda = lda_preds == 1, qda = qda_preds == 1, knn = knn_pred == 1)

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, 1, 0) #for each case: if the majority of algorithms predict B,then the ensemble algorithm predicts B.
mean(ensemble_preds == test_set$Y)

models <- c("linear model", "LDA", "QDA", "K nearest neighbors","Ensemble")
accuracy <- c(mean(linear_preds == test_set$Y),
              mean(lda_preds == test_set$Y),
              mean(qda_preds == test_set$Y),
              mean(knn_pred == test_set$Y),
              mean(ensemble_preds == test_set$Y))

data.frame(Model = models, Accuracy = accuracy)






























