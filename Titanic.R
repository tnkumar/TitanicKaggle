# Load the train and test datasets to create two DataFrames

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
titanic.train = read.csv(train_url, stringsAsFactors = FALSE, header = TRUE)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
titanic.test = read.csv(test_url, stringsAsFactors = FALSE, header = TRUE)

#combine train and test to facilitate processing
titanic.train$IsTrainSet <- TRUE
titanic.test$IsTrainSet <- FALSE
titanic.test$Survived <- NA
titanic.full <- rbind(titanic.train, titanic.test)

#replace missing values of Embarked with S
titanic.full[titanic.full$Embarked=='', "Embarked"] <- 'S'

#categorical casting except for column Survived
titanic.full$Pclass <- as.factor(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)

#clean missing values of age

#identify and remove the outliers
age.upper.whisker <- boxplot.stats(titanic.full$Age)$stats[5]
age.outlier.filter <- titanic.full$Age < age.upper.whisker
titanic.full[age.outlier.filter,]

#create a model for age using parameters Pclass, Sex, SibSp, Parch and Embarked
age.equation = "Age ~ Pclass + Sex + SibSp + Parch + Embarked"
age.model <- lm(
  formula = age.equation,
  data = titanic.full[age.outlier.filter,]
)

age.row <- titanic.full[
  is.na(titanic.full$Age),
  c("Pclass", "Sex", "SibSp", "Parch", "Embarked")
  ]

#predict the age using the model
age.predictions <- predict(age.model, newdata = age.row)
titanic.full[is.na(titanic.full$Age), "Age"] <- age.predictions

#clean missing values of fare
#fare.median <- median(titanic.full$Fare, na.rm = TRUE)
#titanic.full[is.na(titanic.full$Fare), "Fare"] <- fare.median

fare.upper.whisker <- boxplot.stats(titanic.full$Fare)$stats[5]
fare.outlier.filter <- titanic.full$Fare < fare.upper.whisker
titanic.full[fare.outlier.filter,]

fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare.model <- lm(
  formula = fare.equation,
  data = titanic.full[fare.outlier.filter,]
)

fare.row <- titanic.full[
  is.na(titanic.full$Fare),
  c("Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked")
  ]

fare.predictions <- predict(fare.model, newdata = fare.row)
titanic.full[is.na(titanic.full$Fare), "Fare"] <- fare.predictions

# split dataset back out to train and test
titanic.train<- titanic.full[titanic.full$IsTrainSet==TRUE,]
titanic.test<- titanic.full[titanic.full$IsTrainSet==FALSE,]

titanic.train$Survived <- as.factor(titanic.train$Survived)

survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survived.formula <- as.formula(survived.equation)

install.packages("randomForest")
library(randomForest)

titanic.model <- randomForest(formula = survived.formula, data = titanic.train, ntree =500, mtry = 3, nodesize = 0.01 * nrow(titanic.train))

features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
Survived <- predict(titanic.model, newdata = titanic.test)

PassengerId <- titanic.test$PassengerId
output.df <- as.data.frame(PassengerId)
output.df$Survived <- Survived

write.csv(output.df, file="kaggle_submission.csv", row.names = FALSE)



