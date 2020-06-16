
---
title: '"Using Data Mining To Predict Secondary School Student Performance"'
author: "Minchan Shi, Beibhinn Gallagher"
date: "4/19/2020"
output: word_document

This research aims to predict Student Academic Success using linear discriminant analysis, a generalized linear model and classification techniques to analyze the factors that determine the students’ final grades.( Programming Languages: R)
Used Decision trees for math and Portuguese correctly classify outcomes 78% and 96% respectively.
The Generalized Linear model performed in predicting student outcome, with an average misclassification rate of 2.34%.

---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(car)
library(MASS)
library(broom)
library(plyr)
library(dplyr)
library(ggplot2)
library(DT)
library(psych)
library(purrr)
library(pscl)
library(ROCR)
library(e1071)
library(boot)
library(ISLR)
library(tree)
library(randomForest)
```


# Abstract

Data mining in the field of education has been a topic of research for many years largely due to its applicability and wide range of interested groups within the field. Our research aims to predict Student Academic Success using linear discriminant analysis, a generalized linear model and classification techniques to analyze the factors that determine the students’ final grades. The Generalized Linear model performed the best in predicting student outcome, with an average misclassification rate of 2.34%. Decision trees for math and portuguese correctly classify outcomes 78% and 96% respectively. The variables found to be most important in predicting student outcome were students age, class absences, work day alcohol consumption. 

# Introduction

Data mining applications in education have great potential to improve learning outcomes for example by identifying students in need of additional resources or identifying the most effective teaching strategies (Osmanbegovic and Suljic, 2012). Using educational, demographic and social information about 649 portugese students we will do a comparative analysis of regression and classification models to accurately predict students final grades in mathematics and portugese language classes and identify what factors may affect that outcome. As students ourselves, we hope that we may learn something that will help improve our own educational outcomes. 

# Background
Data Mining is the process of modeling relationships and patterns across large amounts of data and has been used extensively in business, science and also education, where it has a wide range of applications for  students, teachers, and administrators (Brittanica). “Modeling student performance is an important tool for both educators and students, [...] for instance, school professionals could perform corrective measures for weak students”(Cortez and Silva, 2008).  Twenty years ago researchers in Singapore did just that and used data mining techniques to help identify students for remedial classes (Ma et al., 2000). Since then data mining techniques have been commonly applied in education. Some of these previous studies have identified socioeconomic background, sex, and ethnicity as influences on academic performance among students. (Thiele, 2014) Other studies have concluded that previous grades were a stronger predictor of future success than demographic factors (Kotsiantis et al., 2004). 

# Data Description
The dataset contains 33 variables consisting of Mathematics and Portugese language grades, demographic, and other social and educational attributes of 649 Portugese students aged 15-22 collected through questionnaires and school reports. The attributes reported in the dataset are: student's school (school: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira;  student's sex (sex: 'F' - female or 'M' - male ); student's age (in years); student's home address type (address: 'U' - urban or 'R' - rural); family size (famsize: 'LE3' - less or equal to 3 or 'GT3' - greater than 3);  parent's cohabitation status (Pstatus: 'T' - living together or 'A' - apart); mother's education (Medu: 0 - none, 1 - primary education, 2 - 5th to 9th grade, 3 -  secondary education or 4 - higher education);  father's education (Fedu: 0 - none, 1 - primary education, 2 - 5th to 9th grade, 3 -  secondary education or 4 - higher education); mother's job (Mjob: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other'); father's job (Fjob: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other'); reason to choose this school ( reason: close to 'home', school 'reputation', 'course' preference or 'other'); student's guardian (guardian: 'mother', 'father' or 'other'); home to school travel time (traveltime: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour); weekly study time (studytime: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours); number of past class failures (failures: n if 1<=n<3, else 4);  extra educational support (schoolsup: yes or no), family educational support (famsup: yes or no); extra paid classes within the course subject (Math or Portuguese) (paid: yes or no); if student were involved in extra-curricular activities (activities: yes or no); if students attended nursery school (nursery: yes or no); if the student wants to take higher education (higher: yes or no); if the student has  Internet access at home (internet: yes or no); if the student was involved with a romantic relationship (romantic: yes or no); the quality of family relationships (famrel: from 1 - very bad to 5 - excellent); amount of  free time after school (freetime: from 1 - very low to 5 - very high); how often student are going out with friends (goout: from 1 - infrequent to 5 - very  frequent); students workday alcohol consumption (Dalc: from 1 - very low to 5 - very high); weekend alcohol consumption (Walc: from 1 - very low to 5 - very high); current health status (health: from 1 - very bad to 5 - very good); number of school absences (absences: from 0 to 93);  first period grade ( G1: a number from 0 to 20); second period grade (G2: a number from 0 to 20); final grade (G3: a number from 0 to 20)

# Data Pre-Processing

There are two sets of data that we are working with, Math and Portugese. They both measure the same variables, yet  the Math dataset has 395 instances while the Portugese dataset has 649. When the datasets are merged by all applicable variables (excluding grades (G1, G2, G3), absences, and if they paid for extra classes within the course subject) there are 320 instances, or rows. This means that in merging the data between Math and Portugese, we only consider students who have taken both Math and portugese and we exclude 75 instances in which there are inconsistencies between the recorded responses for the variables in the two datasets. For outliers there are a few within our merged dataset of 320 such as a student age of 22 when average is 16.5, 6 students travel over an hour to school when the average travel time is less than 30 minutes, most students have no failures but 4 have 3 past class failures, and average absences in Math is 5 yet the most is 75 absences.

```{r, include=FALSE}
d1=read.table("/Users/beibhinngallagher/Downloads/Student/student-mat.csv",sep=";",header=TRUE)
d2=read.table("/Users/beibhinngallagher/Downloads/Student/student-por.csv",sep=";",header=TRUE)
   
dataset = merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet",
                           "guardian", "famsize", "famrel",  "traveltime", "studytime", "failures", "schoolsup", "famsup", "activities",
                           "higher", "romantic", "freetime", "goout", "Walc","Dalc" ,"health"))
# Rename
dataset$absences.math = dataset$absences.x
dataset$G1.math = dataset$G1.x
dataset$G2.math = dataset$G2.x
dataset$G3.math = dataset$G3.x
dataset$paid.math = dataset$paid.x
dataset$absences.por = dataset$absences.y
dataset$G1.por = dataset$G1.y
dataset$G2.por = dataset$G2.y
dataset$G3.por = dataset$G3.y
dataset$paid.por = dataset$paid.y
dataset = subset(dataset, select = -c(absences.x,G1.x, G2.x, G3.x, paid.x,absences.y,G1.y ,G2.y, G3.y, paid.y) )
```

# Descriptive Analytics for Math and Portuguese Data

## Descriptive Analytics for Math
```{r, echo=FALSE}
#Math 
#summary(dataset$G3.math)
print("Standard Deviation")
sd(dataset$G3.math)
print("Variance")
var(dataset$G3.math)
hist(dataset$G3.math,50,
     main = "Distribution of Final Math Grades",
     xlab = "Math Grades")
print("Skewness")
skewness(dataset$G3.math,na.rm = TRUE)
print("Kurtosis")
kurtosis(dataset$G3.math,na.rm = TRUE)
qqPlot(dataset$G3.math, ylab = "Final Math Grades", main = "Q-Q Plot")
#logmath<-log(dataset$G3.math+1)
#qqPlot(logmath)
#hist(logmath,50)
#shapiro.test(logmath)
#shapiro.test(dataset$G3.math)
#create a new dateset inluding grade classification
grade1<-dataset %>%
   filter(!is.na(G3.math))%>%
   mutate(grade=case_when(dataset$G3.math>=15~"Good",
                                dataset$G3.math>=12~"Satisfactory",
                                dataset$G3.math>=10~"Sufficient",
                                dataset$G3.math<10~"Fail")) 
#bar chart to display the math grade classification
ggplot(grade1,aes(x=grade,y=G3.math,color=grade))+geom_bar(stat="identity")+ 
theme_minimal()+labs(title ="Classification of Mathematical Achievement")
```
Visually the Histogram and Q-Q plot suggest the Final Math Grades may have a non-normal distribution. 
Final math grades (G3.math) exhibits skewness of `r skewness(dataset$G3.math,na.rm = TRUE)` indicating a negative skew and kurtosis of `r kurtosis(dataset$G3.math,na.rm = TRUE)` and so is platykurtic. The average final Math grade is `r mean(dataset$G3.math)`, with a variance of `r var(dataset$G3.math)` and a standard deviation of `r sd(dataset$G3.math)`. The Minimum Grade = 0，Median grade = 11，and Maximum grade = 20.
A Q-Q plot of the final Math grades shows some devaition from the norm in the variance of the residuals particularly towards the tails. The Bar chart shows The number of final math outcomes. First, we add a column in the database, indicating final math score from 0 to 20, with a score below 9 and we set it to "Fail", between 10-12 are "Sufficient", between 12-14 is "Satisfactory", and higher than 15 is "Good". The chart is classified by grade. "Fail" refers to the failure of students in math class, but it accounts for the lowest proportion. "Good" means that of all the counts, most students do better in math. 

## Descriptive Analytics for Portuguese
```{r, echo=FALSE}
#Final Portuguese Grades
#summary(dataset$G3.math)
print("Standard Deviation")
sd(dataset$G3.por)
print("Variance")
var(dataset$G3.por)
hist(dataset$G3.por,50,
     main = "Distribution of Final Portuguese Grades",
     xlab = "Portuguese Grades")
print("Skewness")
skewness(dataset$G3.por,na.rm = TRUE)
print("Kurtosis")
kurtosis(dataset$G3.por,na.rm = TRUE)
qqPlot(dataset$G3.por, ylab = "Final Portuguese Grades", main = "Q-Q Plot")
#create a new dateset inluding grade classification
grade1<-dataset %>%
   filter(!is.na(G3.por))%>%
   mutate(grade=case_when(dataset$G3.por>=15~"Good",
                                dataset$G3.por>=12~"Satisfactory",
                                dataset$G3.por>=10~"Sufficient",
                                dataset$G3.por<10~"Fail")) 
#bar chart to display the por grade classification
ggplot(grade1,aes(x=grade,y=G3.por,color=grade,))+geom_bar(stat="identity")+ 
theme_minimal()+labs(title ="Classification of Portuguese Achievement") 
```

The average final Portuguese grade is `r mean(dataset$G3.por)`, with a variance of `r var(dataset$G3.por)` and a standard deviation of `r sd(dataset$G3.por)`. The minimum Portuguese grade is `r min(dataset$G3.por)`, Median grade is `r median(dataset$G3.por)`and maximum grade is `r max(dataset$G3.por)`
The distribution of final portuguese grades (G3.por) exhibits skewness of `r skewness(dataset$G3.por,na.rm = TRUE)` indicating a negative skew and kurtosis of `r kurtosis(dataset$G3.por,na.rm = TRUE)` and so is leptokurtic, which can be seen in the histogram. A Q-Q plot of the final Portuguese grades shows some devaition from the norm in the variance of the residuals. Our classification for the portuguese final grades is the same for the final math grades with with a score below 9 and we set it to "Fail", between 10-12 are "Sufficient", between 12-14 is "Satisfactory", and higher than 15 is "Good". For Portuguese we can see from the bar chart the a very small fraction of the overall student fail portuguese and a majority perform satisfactorily or better.


## Formal Normality Testing for Math and Portuguese Data
To formally assess the normality of our final math and portuguese grades we will prefrom the Shapiro-Wilks test as well as the Kolmogorov-Smirnov test. 
Our Formal Hypothoses are as follows:

K-S test:
H0: The data comes from a normal distribution
HA: At least one value is non-normal

Shapiro-Wilks test:
H0: The data comes from a normal distribution
HA: The data does not coe from a normal distribution
```{r, echo=FALSE}
#KS tests
ks.test(dataset$G3.por, "pnorm")
ks.test(dataset$G3.math, "pnorm")
#Shapiro-Wilks test
shapiro.test(dataset$G3.por)
shapiro.test(dataset$G3.math)
```


In both the cases of the KS test and Shapiro-Wilks test we can reject the null hypotheses and concluded that the distribution of final grades in both Math and Portuguese are non-normal. 

It is possible by transforming our data in some way we can get our final grades into a more normal distribution, however neither a square or log transformation is able to accomplish this. For the remainder of the analysis we will use untransformed data and conclude that our Final Grades in both Portuguese and Math come from non-normal distributions.


# Decision Trees

## Math Decision Tree 
```{r, echo=FALSE}
# Decision tree analysis and predict for math final fail or pass 
set.seed(1)
high<-ifelse(dataset$G3.math>=9,"Pass","Fail")
dataset<-data.frame(dataset,high)
tree.model<- tree(high~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.math,dataset)
summary(tree.model)
plot(tree.model)
text(tree.model,pretty=0)
train<-sample(nrow(dataset),nrow(dataset)/2)
test<-dataset[-train,]
high.test<- high[-train]
tree.model1<- tree(high~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.math,dataset,subset = train)
prediction<-predict(tree.model1,test,type = "class")
table(prediction,high.test)
(7+111)/(7+14+28+111)
#We can prune the tree to prevent overfitting.
set.seed (1)
cv<-cv.tree(tree.model1,FUN=prune.misclass)
plot(cv$size, cv$dev, type = "b")
purn.4<- prune.misclass(tree.model1,best = 4)
tree.pred4<-predict(purn.4,test,type = "class")
table(tree.pred4,high.test)
(4+120)/(4+5+31+120)
purn.2<- prune.misclass(tree.model1,best = 2)
tree.pred2<-predict(purn.2,test,type = "class")
table(tree.pred2,high.test)
(1+125)/(1+34+125)
purn.6<- prune.misclass(tree.model1,best = 6)
tree.pred1<-predict(purn.6,test,type = "class")
table(tree.pred1,high.test)
(6+117)/(6+8+29+117)
```
### Math Decision Tree Analysis
The variables included in the model were 'age', 'absences', 'goout', 'medu', 'walc', 'famrel', and 'health'. In the model, We predicted that about 21 students would Fail math and 139 students would Pass math. The prediction accuracy was close to 74%.We can prune the tree to prevent overfitting,and we see that the training error rate is around 16.5%. The most important indicator of math grade appears to be 'age', followed by 'absences'. In order to properly evaluate the performance of the classification tree on the data, we estimate the test error on the test data to evaluate its performance, this approach leads to correct predictions for 71.5% in the test data set. The pruning process improved the classification accuracy to 78.75% when best=2. 


## Portuguese Decision Tree 
```{r}
# Decision tree analysis and predict for por final fail or pass 
set.seed(1)
high<-ifelse(dataset$G3.por>=9,"Pass","Fail")
dataset<-data.frame(dataset,high)
tree.model<- tree(high~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por,dataset)
summary(tree.model)
plot(tree.model)
text(tree.model,pretty=0)
train<-sample(nrow(dataset),nrow(dataset)/2)
test<-dataset[-train,]
high.test<- high[-train]
tree.model1<- tree(high~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por,dataset,subset = train)
prediction<-predict(tree.model1,test,type = "class")
summary(prediction)
table(prediction,high.test)
(0+145)/(0+145+11+4)
#We can prune the tree to prevent overfitting.
set.seed (1)
cv<-cv.tree(tree.model1,FUN=prune.misclass)
plot(cv$size, cv$dev, type = "b")
set.seed (1)
purn.2<- prune.misclass(tree.model1,best = 2)
tree.pred2<-predict(purn.2,test,type = "class")
table(tree.pred2,high.test)
(0+155)/(0+155+4+1)
set.seed (1)
purn.6<- prune.misclass(tree.model1,best = 7)
tree.pred1<-predict(purn.6,test,type = "class")
table(tree.pred1,high.test)
(0+144)/(0+144+12+4)
purn.4<- prune.misclass(tree.model1,best = 8)
tree.pred4<-predict(purn.4,test,type = "class")
table(tree.pred4,high.test)
(0+145)/(0+145+11+4)
```
## Portuguese Decision Tree Analysis
The variables used in the decision tree model were 'age', 'absences', 'failures', 'fedu', 'medu', 'goout', 'walc' and 'freetime'. In the model, We predicted that 11 students would Fail portuguese and 149 students would Pass portuguese. The prediction accuracy was close to 90.6% and we see that the training error rate is around 15.9%. We can prune the tree to prevent overfitting and the pruning process improved the classification accuracy to 96.8% when best=2. Once again we can see 'age' as the most important variable followed by 'absences'. One possibility for the high classification accuracy is because the proportion of students who fail is so low over-all for portuguese, that it is less likely for the model to wrongly predict a failing classification. This is further discussed in our limitations and future research section.

# Further Research and Analysis for Portuguese data:
After the success of the Classification Tree for the Portuguese dataset, it was decided that our research would continue with the Portuguese dataset. 

#Linear Regression
Our data violates the principal assumption that the response variable comes from a normal distribution and therefor we concluded that Linear Regression will not be effective at predicting final grades for Math or Portuguese. As discussed in our proposal, our preliminary analysis of Linear Regression only explained about 20% of variance and therefore it was not a strong model for our research. 

# Generalized Linear Model
As a linear regression is not suited to our dataset, we believe a Generalized Linear Model may be able to predict the grade outcome in the Portuguese data. 
```{r}
df = subset(dataset, select = -c(absences.math,G1.math, G2.math, G3.math, paid.math) )
```

```{r, echo=FALSE}
set.seed(1)
Pass<-ifelse(df$G3.por>=9,"Pass","Fail")
df<-data.frame(df,Pass)
df
print(" Fit GLM model ")
set.seed(1)
model = glm(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, family = "binomial")
summary(model)
print("Test error Run 1:")
train = sample(dim(df)[1], dim(df)[1] / 2)
model = glm(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, family = "binomial")
probablility = predict(model, newdata = df[-train, ], type = "response")
predict = rep("Fail", length(probablility))
predict[probablility > 0.5] <- "Pass"
glm.m1 = mean(predict != df[-train, ]$Pass)
glm.m1 
print("Test error Run 2:")
train = sample(dim(df)[1], dim(df)[1] / 2)
model = glm(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, family = "binomial")
probablility = predict(model, newdata = df[-train, ], type = "response")
predict = rep("Fail", length(probablility))
predict[probablility > 0.5] <- "Pass"
glm.m2 = mean(predict != df[-train, ]$Pass)
glm.m2
print("Test error Run 3:")
train = sample(dim(df)[1], dim(df)[1] / 2)
model = glm(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, family = "binomial")
probablility = predict(model, newdata = df[-train, ], type = "response")
predict = rep("Fail", length(probablility))
predict[probablility > 0.5] <- "Pass"
glm.m3 = mean(predict != df[-train, ]$Pass)
glm.m3
print("Test error Run 4:")
train = sample(dim(df)[1], dim(df)[1] / 2)
model = glm(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, family = "binomial")
probablility = predict(model, newdata = df[-train, ], type = "response")
predict = rep("Fail", length(probablility))
predict[probablility > 0.5] <- "Pass"
glm.m4 = mean(predict != df[-train, ]$Pass)
glm.m4
print("GLM average test error rate")
M = c(glm.m1 ,glm.m2,glm.m3,glm.m4)
ME = mean(M)
ME
print("GLM average accuracy")
1-ME
```
## Portuguese GLM Analysis
From our Generalized Linear Model we can see that 'failures' has a significant negative effect on the students outcome with a coefficient estimate of -1.112 and a p-value of 2.84e-09 and 'Dalc' also has a significant negative effect with a coefficient estimate of -1.259 and a p-value of 0.00407. We then used the model generated on the training set to predict outcomes for our validation set and calculated the test error rate, or the percentage of time the model misclassified an outcome as compared to the observed results. We repeated this process four times and got an average of 2.34%. Over all, our Generalized Linear Model seems to preform well in classifiying the dataset with an average accuracy rate of 97.65%  

# Linear Discriminant Analysis
We also decided to run a Linear Discriminant Analysis to see if it would perform better than the Generalized Linear Model in predicting the grade outcome for the Portuguese data. 
```{r, echo=FALSE}
print(" Fit LDA model ")
set.seed(1)
lda.fit<- lda(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, subset = train)
summary(lda.fit)
lda.fit
print("Test error Run 1:")
train = sample(dim(df)[1], dim(df)[1] / 2)
lda.fit<-lda(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, subset = train)
lda.pred=predict(lda.fit, newdata=df[-train, ],type = "response")
lda.class=lda.pred$class
pred.lda <- rep("Fail", length(lda.pred))
pred.lda[pred.lda > 0.5] <- "Pass"
lda.m1 = mean(lda.class!=df[-train, ]$Pass)
lda.m1
print("Test error Run 2:")
train = sample(dim(df)[1], dim(df)[1] / 2)
lda.fit<-lda(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, subset = train)
lda.pred=predict(lda.fit, newdata=df[-train, ],type = "response")
lda.class=lda.pred$class
pred.lda <- rep("Fail", length(lda.pred))
pred.lda[pred.lda > 0.5] <- "Pass"
lda.m2 = mean(lda.class!=df[-train, ]$Pass)
lda.m2
print("Test error Run 3:")
train = sample(dim(df)[1], dim(df)[1] / 2)
lda.fit<-lda(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, subset = train)
lda.pred=predict(lda.fit, newdata=df[-train, ],type = "response")
lda.class=lda.pred$class
pred.lda <- rep("Fail", length(lda.pred))
pred.lda[pred.lda > 0.5] <- "Pass"
lda.m3 = mean(lda.class!=df[-train, ]$Pass)
lda.m3
print("Test error Run 4:")
train = sample(dim(df)[1], dim(df)[1] / 2)
lda.fit<-lda(Pass~ age + Medu + Fedu + famrel + studytime+ failures + freetime+ goout+ Walc + Dalc +health + absences.por, data = df, subset = train)
lda.pred=predict(lda.fit, newdata=df[-train, ],type = "response")
lda.class=lda.pred$class
pred.lda <- rep("Fail", length(lda.pred))
pred.lda[pred.lda > 0.5] <- "Pass"
lda.m4 = mean(lda.class!=df[-train, ]$Pass)
lda.m4
print("LDA average test error rate")
M2 = c(lda.m1 ,lda.m2,lda.m3,lda.m4)
ME2 = mean(M2)
ME2
print("LDA average accuracy")
1-ME2
```
## Portuguese LDA Analysis
Here we split our dataset again into training and validation sets and fit a Linear Discriminant Analysis model to our training set. The probabilities of the groups are 0.0375 for fail and 0.9625 for pass meaning 3.75% of our training data represents students who failed and 96.25% represents the students who passed portuguese. Applying a 50% threshold to the posterior probabilities, we used our model to predict outcomes from our validation set four times. Over all, our Linear Discriminant Analysis seems to preform well in classifiying the dataset with an average accuracy rate of 95.78%, but not as well as our Generalized Linear Model.


# Discussion 
Though our Math Decision Tree had a fairly accurate model  with classification accuracy to 78.75% when best=2, we wanted to research if the Portuguese class could result in a better prediction model. This research began with looking at the un-merged dataset including all original 649 Portugese students. This resulted in fairly accurate models for GLM, LDA and QDA with accuracies of 89.31%, 88.54%, and 86.85% respectively. However, we started our research and analysis with the merged dataset to analize the Math class, and subsequently used the merged dataset to compare the Portuguese Decision Tree to our original Math Descision Tree. To keep our research and discussions consistent, the GLM and LDA models were modified to use the merged Portuguese class data as well. The QDA model no longer was viable for this data becuase the dataset was not large enough so it was removed from our analysis. To our suprise, the merged dataset produced better models increasing the GLM and LDA accuracies to 97.66% and 95.78% respectively. This was likely due to the merging process, which was able to rid our data of inconsistencies between the Math and Portuguese data. This was able to provide our models with more reliable data and produced better results. 

# Conclusion:
A decision tree for Final math scores was able to correctly classify student outcome 78% of the time with 'age' being the most important variable followed by 'absences'. The decision tree for Portuguese Grades was able to successfully classify outcomes 96% of the time with 'age' as the most important variable followed by 'absences' as well. Overall, we found that due to the non-normal distributution of final grades, the Generalized Linear model performed the best in predicting student outcome for the Portuguese class, with an average misclassification rate of 2.34% or 97.66% accuracy and identified 'absences' and 'Dalc' as significant variables. Though, not as well performing as Generalized Linear model, our Linear Discriminant Analysis model was successful with an average accuracy rate of 95.78%. Through our research, it was discovered that some of the most influential variables for the final grades are age and class absences, as seen in the classification trees for both Math and Portuguese, and prior failures and weekday alcohol consumption as seen from the GLM model.

# Limitations / Future Research
One limitation of the study is the relatively small sample size and geographic spread. There may be an unseen cultural influence on the factors that we predict student's academic performance that would affect accuracy of our model on students in a different country. When taking a closer look at the Portuguese pruned classification trees, it can be seen in the matrix that the prediction and test for accurate fail predictions (fail, fail) are always 0. It came to our attention that our model for predicting Portuguese grades could be so successful becuase such a small amount of students actually fail the course, unlike Math, where there is a larger portion of students who fail and a less accurate tree model. Looking at this, it would be interesting in future research to expand the analysis for Math as we did for Portuguese and look into a GLM and LDA model as well. In the future, we would also like to use the classifications from the descriptive bar graphs as the predicted classifications rather than the Pass/Fail that we chose to focus on. Another consideration for future research would be to look into more of our available data by creating numerical values for categorical data to expand the variables included in all of the models. We could also look into predicting another outcome such as School, Family size, Relationship status, etc. to see the other insights that this data could provide. Data for future research should aim to increase the amount of students surveyed as well as geographic range of the survey to allow for better statisitcal accuracy and wider applicability. 

#References: 

Clifton, Christopher. “Data Mining.” Encyclopædia Britannica, Encyclopædia Britannica, Inc., 20 Dec. 2019, www.britannica.com/technology/data-mining.

Kotsiantis S.; Pierrakeas C.; and Pintelas P., 2004. Predicting Students’ Performance in Distance Learning Using Machine Learning Techniques. Applied Artificial Intelligence (AAI), 18, no. 5, 411–426.

Ma Y.; Liu B.; Wong C.; Yu P.; and Lee S., 2000.Targeting the right students using data mining. InProc. of 6th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Boston, USA, 457–464.

Osmanbegovic, Edin; Suljic, Mirza (2012) : Data Mining Approach for Predicting Student Performance, Economic Review: Journal of Economics and Business, ISSN 1512-8962, University of Tuzla, Faculty of Economics, Tuzla, Vol. 10, Iss. 1, pp. 3-12

P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

Thiele, Tamara. “Predicting Students' Academic Performance Based on School and Socio-Demographic Characteristics.” Taylor & Francis, 27 Nov. 2014, www.tandfonline.com/doi/full/10.1080/03075079.2014.974528.






