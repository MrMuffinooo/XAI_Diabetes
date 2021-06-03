library(DALEX)
library(DALEXtra)
library(OpenML)
library(mlr)
library(pROC)
library(ggplot2)
library(reshape2)
library(knitr)
library(naniar)
library(visdat)
library(mice)

set.seed(13)

diabetes <- read.csv("diabetes.csv")

diabetes$class[diabetes$class == "tested_positive"] <- 1
diabetes$class[diabetes$class == "tested_negative"] <- 0
diabetes_NA<-diabetes
diabetes_NA[diabetes_NA == 0] <- NA
diabetes_NA$preg[is.na(diabetes_NA$preg)] <- 0
diabetes_NA$insu[is.na(diabetes_NA$insu)] <- 0
diabetes_NA$class[is.na(diabetes_NA$class)] <- 0

m <- sample(1:nrow(diabetes_NA), 0.7*nrow(diabetes_NA))
diabetes_train <- diabetes_NA[m,]
diabetes_test <- diabetes_NA[-m,]
diabetes_train$class <- as.factor(diabetes_train$class)

# train
diabetes_train_skip<-diabetes_train[,-c(3,4,5)]
imp_train2 <- mice(diabetes_train_skip, method = "pmm", m = 1, maxit = 1, nnet.MaxNWts=3000)
diabetes_train_skip <- mice::complete(imp_train2)

#test
diabetes_test_skip<-diabetes_test[,-c(3,4,5)]
imp_test2 <- mice(diabetes_test_skip, method = "pmm", m = 1, maxit = 1, nnet.MaxNWts=3000)
diabetes_test_skip <- mice::complete(imp_test2)

# displaying dataset dimensions
dim(diabetes_train_skip)
dim(diabetes_test_skip)

ggplot(data=diabetes_train_skip, aes_string("class")) +
  geom_bar(fill='darkred') +
  ylab('')

ggplot(data=diabetes_test_skip, aes_string("class")) +
  geom_bar(fill='blue') +
  ylab('')

## RANGER
classif_task_ranger <- makeClassifTask(id = "ranger_tune_random", data = diabetes_train_skip, target = "class")
classif_lrn_ranger <- makeLearner("classif.ranger", predict.type = "prob", par.vals = list(num.trees=776, mtry=1, min.node.size=8, splitrule="extratrees"))
model_skip_ranger<- mlr::train(classif_lrn_ranger, classif_task_ranger)
pred_ranger <- predict(model_skip_ranger, newdata = diabetes_test_skip)$data
explainer_ranger <- explain(id= 'ranger', model = model_skip_ranger,
                     data = diabetes_test_skip[,-6],
                     y = as.numeric(as.character(diabetes_test_skip$class)))
measureAUC(probabilities = pred_ranger$prob.1,
           truth = pred_ranger$truth,
           negative="0",
           positive = "1")
measureFPR(as.factor(pred_ranger$truth),
           pred_ranger$response,
           negative="0",
           positive = "1")
measureBAC(as.factor(pred_ranger$truth),
           pred_ranger$response)

cm_ranger<-caret::confusionMatrix(as.factor(pred_ranger$truth),
                       pred_ranger$response, positive="1")


samp <- sample(1:231, 5)
pp_shap_1<-predict_parts(explainer, new_observation = diabetes_test_skip[samp[2], -6], type = "shap", B = 5)
plot(pp_shap_1)


## ADA
classif_task_ada <- makeClassifTask(id = "ada_tune_random", data = diabetes_train_skip, target = "class")
classif_lrn_ada <- makeLearner("classif.ada", predict.type = "prob", par.vals = list(loss='logistic', type='discrete', iter=81, max.iter=3, minsplit=45, minbucket=4, maxdepth=1))
model_skip_ada<- mlr::train(classif_lrn_ada, classif_task_ada)
pred_ada <- predict(model_skip_ada, newdata = diabetes_test_skip)$data
# explainer_ada <- explain(id='Ada', model = model_random_skip_ada,
#                          data = diabetes_test_skip[,-6],

measureAUC(probabilities = pred_ada$prob.1,
           truth = pred_ada$truth,
           negative="0",
           positive = "1")
measureFPR(pred_ada$truth,
           pred_ada$response,
           negative="0",
           positive = "1")
measureFNR(pred_ada$truth,
           pred_ada$response,
           negative="0",
           positive = "1")
measureBAC(pred_ada$truth,
           pred_ada$response)


caret::confusionMatrix(as.factor(pred_ada$truth),
                       pred_ada$response, positive="1"
                         )


library(ggplot2) 
diabetes_skip<-rbind(diabetes_train_skip, diabetes_test_skip)
ggplot(diabetes_skip, aes(x=mass)) + 
  geom_boxplot(fill="#99d8c9", width=3)+
  ylim(-5,5)+
  theme_minimal()

