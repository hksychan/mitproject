library(stringr)
library(tidyquant)
library(tm)
library(SnowballC)
library(broom)
library(pdftools)
library(tau)
library(wordcloud)
library(dplyr)
library(maxent)
library(RTextTools)
library(mltools)
library(tidyr)
library(dplyr)
library(e1071)

xylabel = read.csv('training_h1_raw_projectD.csv', header = TRUE)
# > colnames(xylabel)
# [1] "X"       "id"      "article" "cat"  

colnames(xylabel)[1] <- c("X")

hot_vec <- function(vec) {
  vec <- as.integer(vec)
  vec[!is.na(vec)] <- 1
  vec[is.na(vec)] <- 0
  return(vec)
}

# The following line might produce memory exhuast error
# If that's the case, try c('id') instead of c("id", "article")
dummy1 <- xylabel[,3]
dummy1 <- as.data.frame(dummy1)
dummy1$id <- xylabel[,2]
dummy2 <- xylabel[,1:2]
dummy2$cat <- xylabel[,4]
test <- reshape(dummy2, direction = "wide", idvar = c("id") , timevar = "cat")
test$X.GCAT <- hot_vec(test$X.GCAT)
test$X.CCAT <- hot_vec(test$X.CCAT)
test$X.MCAT <- hot_vec(test$X.MCAT)
test$X.ECAT <- hot_vec(test$X.ECAT)

dummy3 <- distinct(dummy1)
test$article <- dummy3$dummy1

test_clean <- test[,6:10]

# > colnames(test)
# [1] "id"      "article" "X.ECAT"  "X.GCAT"  "X.CCAT"  "X.MCAT" 
# last four columns are hot vectors, 1s indicate that the article is under this topic, 0 otherwise

# 1.Creating a matrix
doc_matrix <- create_matrix(as.character(test_clean$article),
                            language="english",
                            removeNumbers=TRUE,
                            stemWords=TRUE,
                            removeSparseTerms=0.999,
                            weighting = weightTf)

# 2.Creating a container
stamp1 <- round(dim(doc_matrix)[1]*0.9) # last sample in the trainning set
stamp2 <- stamp1 + 1 # first sample in the test set
stamp3 <- dim(doc_matrix)[1] # last data sample

# create container for each of the topic's labeling
container1 <- create_container(doc_matrix,
                              test_clean$X.GCAT,
                              trainSize=1:stamp1,
                              testSize=stamp2:stamp3,
                              virgin=FALSE)
container2 <- create_container(doc_matrix,
                               test_clean$X.CCAT,
                               trainSize=1:stamp1,
                               testSize=stamp2:stamp3,
                               virgin=FALSE)
container3 <- create_container(doc_matrix,
                               test_clean$X.MCAT,
                               trainSize=1:stamp1,
                               testSize=stamp2:stamp3,
                               virgin=FALSE)
container4 <- create_container(doc_matrix,
                               test_clean$X.ECAT,
                               trainSize=1:stamp1,
                               testSize=stamp2:stamp3,
                               virgin=FALSE)

# 3.Training models for each topic
kernel_name = 'linear'
SVM1 <- train_model(container1,"SVM", kernel = kernel_name, cost = 0.01)
SVM2 <- train_model(container2,"SVM", kernel = kernel_name, cost = 0.1)
SVM3 <- train_model(container3,"SVM", kernel = kernel_name, cost = 0.01)
SVM4 <- train_model(container4,"SVM", kernel = kernel_name, cost = 0.01)

SVM22 <- tune.svm(x = container2@training_matrix,
                     y = container2@training_codes,
                     cost = c(0.01,0.05, 0.1),
                     kernel="linear"
)

summary(SVM22)

# 4.Classifying data using trained models
SVM_CLASSIFY1 <- classify_model(container1, SVM1)
SVM_CLASSIFY2 <- classify_model(container2, SVM2)
SVM_CLASSIFY3 <- classify_model(container3, SVM3)
SVM_CLASSIFY4 <- classify_model(container4, SVM4)


# 5.Analytics
SVM_analytics1 <- create_analytics(container1, SVM_CLASSIFY1)
SVM_analytics2 <- create_analytics(container2, SVM_CLASSIFY2)
SVM_analytics3 <- create_analytics(container3, SVM_CLASSIFY3)
SVM_analytics4 <- create_analytics(container4, SVM_CLASSIFY4)

summary(SVM_analytics1)
summary(SVM_analytics2)
summary(SVM_analytics3)
summary(SVM_analytics4)

# Create your custom ensemble method
# Or write your custom function for calculating the metrics

results <- test$id[stamp2:stamp3]
results$X.GCAT <- as.integar(SVM_CLASSIFY1[,1])
results$X.CCAT <- as.integar(SVM_CLASSIFY2[,1])
results$X.MCAT <- as.integar(SVM_CLASSIFY3[,1])
results$X.ECAT <- as.integar(SVM_CLASSIFY4[,1])

probs <- test$id[stamp2:stamp3]
probs$X.GCAT <- as.numeric(SVM_CLASSIFY1[,2])
probs$X.CCAT <- as.numeric(SVM_CLASSIFY2[,2])
probs$X.MCAT <- as.numeric(SVM_CLASSIFY3[,2])
probs$X.ECAT <- as.numeric(SVM_CLASSIFY4[,2])

