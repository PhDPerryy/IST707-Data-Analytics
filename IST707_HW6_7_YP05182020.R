#Name: Yehuda Perry
#Course: IST707 - Data Analytics
#Date: 05/18/2020
#Assignment: Homework 6-7 - kNN, Random Forest, SVM, Naive Bayes and Decision Tree

#Loading Libraries 
library(datasets)
library(class) 
library(ggplot2)
library(plyr) 
library(dplyr)
library(lattice)
library(caret) 
library(e1071)
library(gmodels)
library(randomForest)
library(tree)
library(MASS) 
library(rpart)
library(rpart.plot)
library(proto)
library(readr)
library(e1071)
library(naivebayes) 
library(statsr)
library(statsExpressions)
library(statsguRu)
library(stats19)
library(tidyverse) 
library(gbm) 
library(rpart.utils)
library(rpart.LAD)
library(Rcpp)
library(Rtsne)
library(RColorBrewer)
library(rattle)
library(lattice)
library(caret)
library(cluster)
library(class)
library(foreign)
library(tree)
library(maptree)

train <- read.csv("~/Downloads/trainkg.csv", header = TRUE, na.strings = c("")) 
test <- read.csv("~/Downloads/testkg.csv", header = TRUE, na.strings = c("")) 

nrow(train)
nrow(test)

#explorer the data
#ETD1
train
test
#ETD2
head(train)
str(train)
head(test)
str(test)
#ETD3
dim(train)
dim(test)


#Disterbution of values - train
hist(train$label, col='orange', breaks = seq(from=-0.5, to=9.5, by=1),
     main="Distribution of Values", xlab='Value')

#Plotting the matrix
flip <- function(matrix){
  apply(matrix, 2, rev)
}
#random sampling
plotdigit<- function(datarow, rm1=F){
  # function will produce an image of the data given
  # the function takes away the first value because it is the target
  title<- datarow[1] # get actual value from training data
  if(rm1){
    datarow<- datarow[-1] # remove the actual value
  }
  datarow<- as.numeric(datarow) # convert to numeric
  x<- rep(0:27)/27 
  y<- rep(0:27)/27
  z<- matrix(datarow, ncol=28, byrow=T)
  rotate <- function(x) t(apply(x, 2, rev))
  z<- rotate(z)
  image(x,y,z, main=paste("Actual Value:", title), col=gray.colors(255, start=1, end=0), asp=1,
        xlim=c(0,1), ylim=c(-0.1,1.1), useRaster = T, axes=F, xlab='', ylab='')
}

#par(mfrow=c(3,4))
set.seed(1)
rows<- sample( 1:42000, size=12)
for(i in rows){
  plotdigit(train[i,],rm1=T)
}

#Data Transformation and Cleansing
#Creating samples
train_sam <- train[seq(1, nrow(train), 10), ]
test_sam <- test[seq(1, nrow(test), 10), ]

#Removing pixels that have 0 in all images
train_clean <- train_sam[, colSums(train_sam != 0) > 0]

#Removing pixels with low variances
all_var <- data.frame(apply(train_clean[-1], 2, var))
colnames(all_var) <- "Variances"

#Sorting variances
all_var <- all_var[order(all_var$Variances), , drop = FALSE] 
#Creating number labels for variances
num_labels <- c(1:661) 
numbered_var <- cbind(all_var,num_labels) 
summary(all_var)

plot(all_var$Variances, type = "l",xlab="Pixel", ylab="Pixel variance", lwd=2)
abline(h=5181, col="red", lwd=2)
plot(all_var$Variances, type = "l", xlim = c(0,400), ylim=c(0,800))
abline(h=300, col="red", lwd=2)

good_var <- subset(all_var, all_var$Variances >= 300, "Variances")
good_var_pixels <- row.names(good_var)
train_clean <- train_clean[, c("label", good_var_pixels)] 

#Normalizing the Data
min_max_func <- function (x) {
  a <- (x - min(x))
  b <- (max(x) - min(x))
  return(a / b)
}

clean_train_nolabel <- train_clean[, -1]
clean_train_nolabel_normalize <- as.data.frame(lapply(clean_train_nolabel, min_max_func))
train_clean <- cbind(label = train_clean$label, clean_train_nolabel_normalize)

head(train_clean)

#Creating Test and Train sets for the Training Data Set
train_clean$label <- as.factor(train_clean$label)
set.seed(123)
idx <- sample(1:nrow(train_clean), size = 0.8 * nrow(train_clean))
train_set <- train_clean[idx, ]
test_set <- train_clean[ -idx, ]

#Building the KNN Model
#Dividing test and train sets into attributes only and labels only
train_set_numonly <- train_set[,-1] 
train_set_labels <- train_set[,1]
test_set_numonly <- test_set[,-1]
test_set_labels <- test_set[,1]

#Choosing the K and building the model
k <- round(sqrt(nrow(train_clean))) 

KNN_Model <- knn(train = train_set_numonly, test = test_set_numonly, 
                 cl= train_set_labels ,k = k, prob=TRUE)

#confusion matrix
confusionMatrix(test_set_labels, KNN_Model) 

#Build the Random Fores Model
RF_Model <- randomForest(label~., test_set)
RF_Pred <- predict(RF_Model, test_set)

#Confusion Matrix and Accuracy
confusionMatrix(test_set$label, RF_Pred) 

#Important Variables
varImpPlot(RF_Model)
plot(RF_Model)
RF_Tree = tree(RF_Model,data=train_set)
plot(RF_Tree)
text(RF_Tree)

#SVM
KT_P = "polynomial" # Polynomial Kernel

SVM <- svm(label~., data=train_set, kernel=KT_P, cost=100, scale=FALSE)
print(SVM)

#SVM Predication
SVM_Pred <- predict(SVM, test_set_numonly , type="class")

#Confusion Matrix
confusionMatrix(test_set_labels, SVM_Pred) 

#Naive Bayes
NB <- naiveBayes(label~., data=train_set, kernel=KT_P, cost=100, scale=FALSE)
print(NB)

NB_Pred <- predict(NB, test_set_numonly , type="class")

#Confusion Matrix
confusionMatrix(test_set_labels, NB_Pred) 

#Decision Tree

DT <- rpart(train_set$label ~ .,method = "class", data = train_set)
printcp(DT)

DT_Pred <- predict(DT, test_set_numonly , type="class")

#Confusion Matrix
confusionMatrix(test_set_labels, DT_Pred) 

error.rate.rpart <- sum(test_set$label != prediction.rpart)/nrow(test_set)
print(paste0("Accuracy (Precision): ", 1 - error.rate.rpart))

#Visualize Decision Tree
heat.tree <- function(tree, low.is.green=FALSE, ...) { # dots args passed to prp
  y <- DT$frame$yval
  if(low.is.green)
    y <- -y
  max <- max(y)
  min <- min(y)
  cols <- rainbow(99, end=.36)[
    ifelse(y > y[1], (y-y[1]) * (99-50) / (max-y[1]) + 50,
           (y-min) * (50-1) / (y[1]-min) + 1)]
  prp(DT, branch.col=brewer.pal(10,"Set3"), box.col=brewer.pal(10,"Set3"), ...)
}

heat.tree(DT, type=4, varlen=0, faclen=0, fallen.leaves=TRUE)

drawTreeNodes(DT, cex = 0.5, nodeinfo = TRUE, col = gray(0:8/8))
