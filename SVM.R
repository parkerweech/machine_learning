#The first time through, you'll need to install the LIBSVM package:
#install.packages('e1071', dependencies = TRUE);

# Include the LIBSVM package
library(e1071)

# Load our old friend, the Iris data set
# Note that it is included in the default datasets library
library(datasets)
data(iris)

# For your assignment, you'll need to read from a CSV file.
# Conveniently, there is a read.csv() function that can be used like so:
data_1 <- read.csv(file="letters.csv", 
                    head=TRUE, 
                    sep=",")

data_2 <- read.csv(file="vowel.csv", 
                   head=TRUE, 
                   sep=",")

# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows <- 1:nrow(data_1)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

allRows_2 <- 1:nrow(data_2)
testRows_2 <- sample(allRows_2, trunc(length(allRows_2) * 0.3))

# The test set contains all the test rows
data_1_Test <- data_1[testRows,]
data_2_Test <- data_2[testRows_2,]

# The training set contains all the other rows
data_1_Train <- data_1[-testRows,]
data_2_Train <- data_2[-testRows_2,]

# Train an SVM model
# Tell it the attribute to predict vs the attributes to use in the prediction,
#  the training data to use, and the kernal to use, along with its hyperparameters.
#  Please note that "Species~." contains a tilde character, rather than a minus
model <- svm(letter~., data = data_1_Train, kernel = "radial", gamma = 0.001, cost = 10)
model_2 <- svm(Class~., data = data_2_Train, kernel = "radial", gamma = 0.001, cost = 10)

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, data_1_Test[,-1])
prediction_2 <- predict(model_2, data_2_Test[,-13])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = data_1_Test[,1])
confusionMatrix_2 <- table(pred = prediction_2, true = data_2_Test[,13])

# Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == data_1_Test[,1]
agreement_2 <- prediction_2 == data_2_Test[,13]

accuracy <- prop.table(table(agreement))
accuracy_2 <- prop.table(table(agreement_2))

# Print our results to the screen
print(confusionMatrix)
print(accuracy)
print(confusionMatrix_2)
print(accuracy_2)