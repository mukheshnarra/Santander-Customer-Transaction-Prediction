#removing the global enviornment variables
rm(list=ls())

#setting working directory
setwd('D:/Edwisor/Santander Customer Transaction')


#loading required libraries
pacman::p_load(fBasics,ggplot2,corrgram,plyr,DMwR,caret,Boruta,xgboost,randomForest,glmnet,rplos,gbm,ROSE,sampling,DataCombine, inTrees,fastDummies,rattle,rBayesianOptimization,tidyverse)
#loading the train data
train_data=read.csv('train.csv')
#loading the test data
test_data=read.csv('test.csv')
#top 5 records of train_data
head(train_data,5)
#Dimension of train_data
dim(train_data)
#Summary of the train data
str(train_data)
#top 5 records of test_data
head(test_data,5)
#Dimension of test_data
dim(test_data)
#Summary of the test_data
str(test_data)

#convert to factor
train_data$target<-as.factor(train_data$target)
#Count of target classes
table(train_data$target)
#Percenatge counts of target classes
table(train_data$target)/length(train_df$target)*100
#Bar plot for count of target classes
plot1<-ggplot(train_data,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')
grid.arrange(plot1)

#Finding the missing values in train data
missing_val_train<-data.frame(missing_val=apply(train_data,2,function(x){sum(is.na(x))}))

#Finding the missing values in test data
missing_val_test<-data.frame(missing_val=apply(test_data,2,function(x){sum(is.na(x))}))

#function  for distribution of train attributes
plot_distribution<-function(low,high){
  for (var in names(train_data)[c(low:high)]){
    target<-train_data$target
    plot<-ggplot(train_data, aes(x=train_data[[var]],fill=target)) +
      geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
    print(plot)
  }
}
#Distribution of train attributes from 3 to 102
plot_distribution(3,102)
#Let us see distribution of train attributes from 103 to 202
plot_distribution(103,202)
#let us see distribution of test attributes from 2 to 101
plot_density(test_df[,c(2:101)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))
#let us see distribution of test attributes from 2 to 101
plot_density(test_df[,c(102:201)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))

#Applying the function to find mean values per row in train and test data.
train_mean<-apply(train_data[,-c(1,2)],MARGIN=1,FUN=mean)
test_mean<-apply(test_data[,-c(1)],MARGIN=1,FUN=mean)
ggplot()+
  #Distribution of mean values per row in train data
  geom_density(data=train_data[,-c(1,2)],aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=test_data[,-c(1)],aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per row',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find mean values per column in train and test data.
train_mean<-apply(train_data[,-c(1,2)],MARGIN=2,FUN=mean)
test_mean<-apply(test_data[,-c(1)],MARGIN=2,FUN=mean)
ggplot()+
  #Distribution of mean values per column in train data
  geom_density(aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per column in test data
  geom_density(aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per column',title="Distribution of mean values per row in train and test dataset")



#Applying the function to find standard deviation values per row in train and test data.
train_sd<-apply(train_data[,-c(1,2)],MARGIN=1,FUN=sd)
test_sd<-apply(test_data[,-c(1)],MARGIN=1,FUN=sd)
ggplot()+
  #Distribution of sd values per row in train data
  geom_density(data=train_data[,-c(1,2)],aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=test_data[,-c(1)],aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per row',title="Distribution of sd values per row in train and test dataset")

#Applying the function to find sd values per column in train and test data.
train_sd<-apply(train_data[,-c(1,2)],MARGIN=2,FUN=sd)
test_sd<-apply(test_data[,-c(1)],MARGIN=2,FUN=sd)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per column',title="Distribution of std values per column in train and test dataset")


#Applying the function to find skewness values per row in train and test data.
train_skew<-apply(train_data[,-c(1,2)],MARGIN=1,FUN=skewness)
test_skew<-apply(test_data[,-c(1)],MARGIN=1,FUN=skewness)
ggplot()+
  #Distribution of skewness values per row in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per row',title="Distribution of skewness values per row in train and test dataset")

#Applying the function to find skewness values per column in train and test data.
train_skew<-apply(train_data[,-c(1,2)],MARGIN=2,FUN=skewness)
test_skew<-apply(test_data[,-c(1)],MARGIN=2,FUN=skewness)
ggplot()+
  #Distribution of skewness values per column in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per column',title="Distribution of skewness values per column in train and test dataset")

#Applying the function to find kurtosis values per row in train and test data.
train_kurtosis<-apply(train_data[,-c(1,2)],MARGIN=1,FUN=kurtosis)
test_kurtosis<-apply(test_data[,-c(1)],MARGIN=1,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per row',title="Distribution of kurtosis values per row in train and test dataset")

#Applying the function to find kurtosis values per column in train and test data.
train_kurtosis<-apply(train_data[,-c(1,2)],MARGIN=2,FUN=kurtosis)
test_kurtosis<-apply(test_data[,-c(1)],MARGIN=2,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per column',title="Distribution of kurtosis values per column in train and test dataset")

#Correlations in train data
#convert factor to int
train_data$target<-as.numeric(train_data$target)
train_correlations<-cor(train_data[,c(2:202)])
train_correlations
#dupllicates train values
unique_values_train=data.frame(sapply(train_data[3:202],function(x) length(unique(x))))
#duplicate values for test values
unique_values_test=data.frame(sapply(test_data[2:201],function(x) length(unique(x))))
#Feature Engineering
#top 5 unique feature
features_unique=c('var_68','var_108','var_12','var_91','var_103')
#function for creating the label_count_encoder
count_encoder<-function(df,features_unique)
{
  for(i in features_unique)
  {
    count=sort(table(df[i]),decreasing = T)
    j<-paste(i,'_label_encoding')
    X=mapvalues(df[,i],from=rownames(count),to=1:length(rownames(count)))   
    X=data.frame(X)
    colnames(X)[1]<-j
    df=cbind(df,X)
  }
  
  return(df)
  
}
df_encode_values=count_encoder(train_data,features_unique)
#converting the target to factor
df_encode_values$target=as.factor(df_encode_values$target)
#creating further balancing the imbalanced data by SMOTE 
imbalanced_data<-SMOTE(target ~ ., data=df_encode_values[,-c(1)],perc.over = 100)
#creating further variables
imbalanced_data['mean']=apply(imbalanced_data[,2:206],MARGIN=1,FUN=mean)
imbalanced_data['max']=apply(imbalanced_data[,2:206],MARGIN=1,FUN=max)
imbalanced_data['min']=apply(imbalanced_data[,2:206],MARGIN=1,FUN=min)
imbalanced_data['std']=apply(imbalanced_data[,2:206],MARGIN=1,FUN=stdev)
imbalanced_data['median']=apply(imbalanced_data[,2:206],MARGIN=1,FUN=median)
imbalanced_data['skew']=apply(imbalanced_data[,2:206],MARGIN=1,FUN=skewness)
imbalanced_data['kurtosis']=apply(imbalanced_data[,2:206],MARGIN=1,FUN=kurtosis)
#feature scaling
scaled_data=data.frame(scale(imbalanced_data[,-c(1)]))
scaled_data=cbind(imbalanced_data[,'target'],scaled_data)
colnames(scaled_data)=colnames(imbalanced_data)
#Splitting the train data into two parts one for validaton and another for training
train_index=createDataPartition(scaled_data$target,p=0.80,list=F)
X_train_data=scaled_data[train_index,]
X_valid_data=scaled_data[-train_index,]

#-------------------------feature importance by boruta--------------------------#

boruta.results<-Boruta(scaled_data[,-c(1)],scaled_data[,'target'],
                       maxRuns=101,
                       doTrace=0)
summary(boruta.results)
getSelectedAttributes(boruta.results)
#---------------------------------------Modeling-------------------------------------------#

#applying the logistic regression
lr<-glm(target~.,data=X_train_data,family='binomial')
summary(lr)
lg_predictions<-predict(lr,X_valid_data[,-c(1)])
roc.curve(X_valid_data[,c(1)],lg_predictions,plotit = TRUE)#0.786
confusionMatrix(table(X_valid_data[,c(1)]),lg_predictions)

#applying the naive 
library(e1071)
NB_model=naiveBayes(target~ .,data=X_train_data)
nb_predictions=predict(NB_model,X_valid_data[,-c(1)],type='class')
roc.curve(X_valid_data[,c(1)],nb_predictions,plotit = TRUE)
nb_confusion_matrix=table(X_valid_data[,'target'],nb_predictions)#0.80

#applying the knn
library(class)
knn_predictions=knn(X_train_data[,-c(1)],X_valid_data[,-c(1)],X_train_data$target,k=3)
#accuracy 
confusionmatrix_knn=table(X_valid_data$target,knn_predictions)
roc.curve(X_valid_data[,c(1)],knn_predictions,plotit = TRUE)#0.5

#applying Random Forest
RFmodel=randomForest(target ~.,X_train_data,importance=TRUE,ntree=400)
RF_predictions=predict(RFmodel,X_valid_data[,-c(1)])
RF_confusionmatrix=table(X_valid_data$target,RF_predictions)
confusionMatrix(RF_confusionmatrix)
roc.curve(X_valid_data[,c(1)],RF_predictions,plotit = TRUE)#0.84

#applying the xgboost
xgboost=xgboost(data=X_train_data[,-c(1)],label=X_train_data[,'target'],max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
xg_predictions=predict(xgboost,X_valid_data[,-c(1)])
confusionMatrix(table(X_valid_data$target,xg_predictions))
roc.curve(X_valid_data[,c(1)],xg_predictions,plotit = TRUE)#0.82



#----------------------------------#Hyperparameter Tuning------------------------------------#
#so furthur moving we can fine tune the random forrest
#where ntree and mtry are varibales used for tuning random forrest
plot(RFmodel)
bmtry<-tuneRF(X_train_data[,-c(1)],X_train_data[,c(1)],ntreeTry = 700,stepFactor = 1.5, improve = 1e-5)
bmtry

#------------------------------after hyperparameter tuning applying Random Forest------------------------------------#
RFmodel=randomForest(target ~.,X_train_data,importance=TRUE,ntree=700,mtry=10)
RF_predictions=predict(RFmodel,X_valid_data[,-c(1)])
RF_confusionmatrix=table(X_valid_data$target,RF_predictions)
confusionMatrix(RF_confusionmatrix)
roc.curve(X_valid_data[,c(1)],RF_predictions,plotit = TRUE)

#--------------------------predictions of test_data-----------------------------#
test_filtered=count_encoder(test_data[,-c(1)],features_unique)
id_Code=data.frame(test[,1])
test_filtered['mean']=apply(test_filtered[,1:200],MARGIN=1,FUN=mean)
test_filtered['max']=apply(test_filtered[,1:200],MARGIN=1,FUN=max)
test_filtered['min']=apply(test_filtered[,1:200],MARGIN=1,FUN=min)
test_filtered['std']=apply(test_filtered[,1:200],MARGIN=1,FUN=stdev)
test_filtered['median']=apply(test_filtered[,1:200],MARGIN=1,FUN=median)
test_filtered['skew']=apply(test_filtered[,1:200],MARGIN=1,FUN=skewness)
test_filtered['kurtosis']=apply(test_filtered[,1:200],MARGIN=1,FUN=kurtosis)
test_after=data.frame(scale(test_filtered))
colnames(test_after)=colnames(test_filtered)
final_predictions=data.frame(predict(RFmodel,test_after),names=c('target'))
final_predictions<-cbind(id_Code,final_predictions)
write.csv(final_predictions,row.names = F)

