
library(e1071)

library(datasets)
data(iris)

data_1 <- read.csv(file="letters.csv", 
                    head=TRUE, 
                    sep=",")

data_2 <- read.csv(file="vowel.csv", 
                   head=TRUE, 
                   sep=",")

allRows <- 1:nrow(data_1)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

allRows_2 <- 1:nrow(data_2)
testRows_2 <- sample(allRows_2, trunc(length(allRows_2) * 0.3))

data_1_Test <- data_1[testRows,]
data_2_Test <- data_2[testRows_2,]

data_1_Train <- data_1[-testRows,]
data_2_Train <- data_2[-testRows_2,]

model <- svm(letter~., data = data_1_Train, kernel = "radial", gamma = 0.001, cost = 10)
model_2 <- svm(Class~., data = data_2_Train, kernel = "radial", gamma = 0.001, cost = 10)

prediction <- predict(model, data_1_Test[,-1])
prediction_2 <- predict(model_2, data_2_Test[,-13])

confusionMatrix <- table(pred = prediction, true = data_1_Test[,1])
confusionMatrix_2 <- table(pred = prediction_2, true = data_2_Test[,13])

agreement <- prediction == data_1_Test[,1]
agreement_2 <- prediction_2 == data_2_Test[,13]

accuracy <- prop.table(table(agreement))
accuracy_2 <- prop.table(table(agreement_2))

print(confusionMatrix)
print(accuracy)
print(confusionMatrix_2)
print(accuracy_2)