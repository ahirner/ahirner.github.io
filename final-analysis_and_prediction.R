library (ggplot2) ; library (caret) ; library (randomForest)

set.seed(1)

setwd("~/Documents/Studium/Other/Practical ML")
pml_training <- read.csv("pml-training.csv", na.strings = c("", "NA", "#DIV/0!"))
pml_testing <- read.csv("pml-testing.csv", na.strings = c("", "NA", "#DIV/0!"))

inTrain <- createDataPartition(y=pml_training$classe, p = 0.7, list=F)

training <- pml_training[inTrain,]
testing <- pml_training[-inTrain,]

training_new <- training[training$new_window == "yes", ]
#clean the rows with too many N/As
training_filter <- training[, colSums(is.na(training)) < nrow(training)*0.95]
training_new <- training_new[, colSums(is.na(training_new)) < nrow(training_new)*0.95]

#2do: what about the rest of the N/As --> try in randomForest algo

featurePlot(x=training_new$max_roll_belt, y=training_new$classe)
#E seems to be quite off in this from others (but other belt values as well, like total_accel_belt)
qplot(training_filter$yaw_arm, training_filter$raw_timestamp_part_2, colour=training_filter$classe)
# some obvious patterns between the different exercises visible

#lets look if there are some obvious characteristics in only the summary statistics (new_window=yes)
training_new_num <- training_new[sapply(training_new, is.numeric)]
training_new_num <- training_new_num[, -nearZeroVar(training_new_num)]
preProc = preProcess.default(x = training_new_num, method = "pca", pcaComp = 2)
training_newPC <- predict(preProc, training_new_num)
plot(training_newPC[,1], training_newPC[,2], col=training_new$classe)
#nope! some other pattern, but not correlated with classe --> try intermediate measurements
training_num <- training[sapply(training, is.numeric)]
training_num <- training_num[, -nearZeroVar(training_num)]
training_num <- subset(training_num, select = -X)
preProc = preProcess.default(x = training_num, method = "pca", pcaComp = 2)
training_PC <- predict(preProc, training_num)
plot(training_PC[,1], training_PC[,2], col=training$classe)
#nope! --> feature selection is critical

features_omit <- c("row.names", "X","user_name", "raw_timestamp_part_1", "cvtd_timestamp", "new_window", "num_window", "classe")
features <- names(training_filter)[!(names(training_filter) %in% features_omit)]
rf <- randomForest(x = training_filter[,features], y = training_filter$classe)
predicted <- predict(rf, testing[, features])
cfm <- confusionMatrix(testing$classe, predicted)

#get the real test predictions for assignment and automatic grading
predict_test <- predict(rf, pml_testing[, features])

#check the importance ex post
imp <- importance(rf)
imp_order <- imp[order(imp[,1], decreasing = T),][1:5]
plot(imp_order, xaxt="n", main = "importance of variables")
axis(1, at = 1: 5, labels = names(imp_order))
#plot the first and third most important features against classe
qplot(training_filter$roll_belt, training_filter$pitch_forearm, colour=training_filter$classe)
