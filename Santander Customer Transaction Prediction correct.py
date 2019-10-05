# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:11:18 2019

@author: MUKHESH
"""

#importing useful libraries
import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
import pandas_profiling as pp
import webbrowser as wb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,StratifiedKFold,train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score,recall_score,precision_score,f1_score,classification_report,confusion_matrix,roc_curve
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
from boruta.boruta_py import BorutaPy 
import joblib

#setting the pandas to view better in IPython Console
pd.options.display.max_columns=10
pd.options.display.max_rows=10
#setting the working directory
os.chdir('D:/Edwisor/Santander Customer Transaction')
#importing the train_data
train_data=pd.read_csv('train.csv')
#importing the test_data
test_data=pd.read_csv('test.csv')
#printing information about the training data
print(train_data.info())
#checking the data types of the variables 
print(train_data.dtypes)
#printing information about the training data
print(test_data.info())
#checking the data types of the variables 
print(test_data.dtypes)

#as we can see apart from id_code and target all are float64 which means 200 variables are numeric variables

#dropping the variable ID_code which does'nt add any information
train_data.drop(columns=['ID_code'],inplace=True)
#dropping in test data also
ID_code=test_data['ID_code']
test_data.drop(columns=['ID_code'],inplace=True)

#From graph we can see that target class 0 is dominating the target class 1 where it is clearly imbalanced data
#Check for any missing values in train_data
Missing_value_train=pd.DataFrame(np.transpose(pd.DataFrame(train_data.isnull().sum(),columns=['count']).reset_index().values))
print(Missing_value_train)
#Check for any missing values in test_data
Missing_value_test=pd.DataFrame(np.transpose(pd.DataFrame(test_data.isnull().sum(),columns=['count']).reset_index().values))
print(Missing_value_test)
 #as we see there is no missing values


#let see the distribution of target varible in train data
sn.countplot(x=train_data.iloc[:,1],data=train_data)

#so let us describe the data to know more about train data
print(train_data.describe())

#so let us describe the data to know more about test data
print(test_data.describe())

#mean values are distributed over a large range. 

#so let we check the distribution of the variable according to the class distribution
features=train_data.columns.values[1:201]
df0=train_data[train_data.target==0]
df1=train_data[train_data.target==1]
def plot_distribution(df0,df1,features,label1,label2):
    i = 0
    sn.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))  
    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sn.distplot(df0[feature], hist=False,label=label1)
        sn.distplot(df1[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
plt.show()
#for target class
plot_distribution(df0,df1,features[0:100],'0','1')
plot_distribution(df0,df1,features[1:200],'0','1')
#for train and test data
plot_distribution(train_data,test_data,features[0:100],'train','test')
plot_distribution(train_data,test_data,features[1:200],'train','test')
#as we can see the most variables distribution for two different class are different
#as we can see that some varibales follows the bivariate distribution which means sum of two vvariables gives the normal distributuion



#as we don't know anything about the data.so we can analyze the data based on row and column basis
#per column train_data and test data
plt.figure(figsize=(16,8))
plt.title("Distribution of mean values per column in the train and test set")
sn.distplot(train_data[features].mean(axis=0),color='red',kde=True,bins=120,label='train')
sn.distplot(test_data[features].mean(axis=0),color='blue',kde=True,bins=120,label='test')
plt.legend()
plt.show()
#per row train_data and test_data
plt.figure(figsize=(16,8))
plt.title("Distribution of mean values per row in the train and test set")
sn.distplot(train_data[features].mean(axis=1),color='red',kde=True,bins=120,label='train')
sn.distplot(test_data[features].mean(axis=1),color='blue',kde=True,bins=120,label='test')
plt.legend()
plt.show()

#per column mean per target
plt.figure(figsize=(16,8))
plt.title("Distribution of mean values per column in the train_data for target classes 0 and 1")
sn.distplot(df0[features].mean(axis=0),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].mean(axis=0),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()
#per row mean per target
plt.figure(figsize=(16,8))
plt.title("Distribution of mean values per row in the train_data for target classes 0 and 1")
sn.distplot(df0[features].mean(axis=1),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].mean(axis=1),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()

#per row max target class
plt.figure(figsize=(16,8))
plt.title("Distribution of max values per row in the train_data for target classes 0 and 1")
sn.distplot(df0[features].max(axis=1),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].max(axis=1),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()
#per column max target class
plt.figure(figsize=(16,8))
plt.title("Distribution of max values per column in the train_data for target classes 0 and 1")
sn.distplot(df0[features].max(axis=0),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].max(axis=0),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()
#per row max test_Data and train_data max
plt.figure(figsize=(16,8))
plt.title("Distribution of max values per row in the train_data and test_data")
sn.distplot(train_data[features].max(axis=1),color='red',kde=True,bins=120,label='train')
sn.distplot(test_data[features].max(axis=1),color='blue',kde=True,bins=120,label='test')
plt.legend()
plt.show()

#per column max test_Data and train_data max
plt.figure(figsize=(16,8))
plt.title("Distribution of max values per column in the train_data and test_data")
sn.distplot(train_data[features].max(axis=0),color='red',kde=True,bins=120,label='train')
sn.distplot(test_data[features].max(axis=0),color='blue',kde=True,bins=120,label='test')
plt.legend()
plt.show()

#per row min target class
plt.figure(figsize=(16,8))
plt.title("Distribution of min values per row in the train_data for target classes 0 and 1")
sn.distplot(df0[features].min(axis=1),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].min(axis=1),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()
#per column min target class
plt.figure(figsize=(16,8))
plt.title("Distribution of min values per column in the train_data for target classes 0 and 1")
sn.distplot(df0[features].min(axis=0),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].min(axis=0),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()
#per row min test_Data and train_data max
plt.figure(figsize=(16,8))
plt.title("Distribution of min values per row in the train_data and test_data")
sn.distplot(train_data[features].min(axis=1),color='red',kde=True,bins=120,label='train')
sn.distplot(test_data[features].min(axis=1),color='blue',kde=True,bins=120,label='test')
plt.legend()
plt.show()

#per column min test_Data and train_data 
plt.figure(figsize=(16,8))
plt.title("Distribution of min values per column in the train_data and test_data")
sn.distplot(train_data[features].min(axis=0),color='red',kde=True,bins=120,label='train')
sn.distplot(test_data[features].min(axis=0),color='blue',kde=True,bins=120,label='test')
plt.legend()
plt.show()


#per column std test_data and train_data
plt.figure(figsize=(16,8))
plt.title("Distribution of std values per column in the train_data and test_data")
sn.distplot(train_data[features].std(axis=0),color='red',kde=True,bins=120,label='train')
sn.distplot(train_data[features].std(axis=0),color='blue',kde=True,bins=120,label='test')
plt.legend()
plt.show()
#per row std test_data and train_data
plt.figure(figsize=(16,8))
plt.title("Distribution of std values per row in the train_data and test_data")
sn.distplot(train_data[features].std(axis=1),color='red',kde=True,bins=120,label='train')
sn.distplot(test_data[features].std(axis=1),color='blue',kde=True,bins=120,label='test')
plt.legend()
plt.show()
#per row skew
plt.figure(figsize=(16,8))
plt.title("Distribution of skew values per row in the train_data for class 0 and 1")
sn.distplot(df0[features].skew(axis=1),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].skew(axis=1),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()
#per column skew
plt.figure(figsize=(16,8))
plt.title("Distribution of skew values per column in the train_data for class 0 and 1")
sn.distplot(df0[features].skew(axis=0),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].skew(axis=0),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()
#per column kurtosis
plt.figure(figsize=(16,8))
plt.title("Distribution of kurtosis values per colummn in the train_data for class 0 and 1")
sn.distplot(df0[features].kurtosis(axis=0),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].kurtosis(axis=0),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()
#per row kurtosis
plt.figure(figsize=(16,8))
plt.title("Distribution of kurtosis values per row in the train_data for class 0 and 1")
sn.distplot(df0[features].kurtosis(axis=1),color='red',kde=True,bins=120,label='0')
sn.distplot(df1[features].kurtosis(axis=1),color='blue',kde=True,bins=120,label='1')
plt.legend()
plt.show()

#let us check unique value in train_data per columns where use in the feature engineering
unique_values_train=[]
for feature in features:
    values=train_data[feature].value_counts()
    unique_values_train.append([feature,values.max(),values.idxmax(),values.shape[0]])
unique_values_train=pd.DataFrame(unique_values_train)
#we can see var_68,var_108,var_126,var_12 where duplicate values count is high 
#check the duplicates values same in test_data also
unique_values_test=[]
for feature in features:
    values=test_data[feature].value_counts()
    unique_values_test.append([feature,values.max(),values.idxmax(),values.shape[0]])
unique_values_test=pd.DataFrame(unique_values_test)
#we can see var_68,var_108,var_126,var_12 same high can see in this in feature engineering
#let us check the correlation between the variables 

correlation_data=pd.DataFrame(train_data.corr().iloc[:,0])
#Overall, the correlation of the features with respect to target are very low.
#lets plot the correlation plot for top values
corr_top=correlation_data.loc[ (correlation_data['target'] < -0.06) | (correlation_data['target'] > 0.05)]
corr_index=list(corr_top.index)
corr_index.remove('target')
correlation_imp_data=train_data[corr_index].corr()
f,ax=plt.subplots(figsize=(20,20))
sn.heatmap(correlation_imp_data,vmax=1., square=True,cmap="YlGnBu",annot=True,linewidths=.5,ax=ax)
plt.show()
#Plotting heatmap is done to identify if there are any strong monotonic relationships between these important features. If the values are high, then probably we can choose to keep one of those variables in the model building process. But, we are doing this only for small set of features. we can even try other techniques to explore other features in the dataset.
#Pair plot for distinction of variable data
sn.pairplot(train_data.iloc[:,0:10],hue='target') 
   
#Feature Sclaing 
#scaling the feature into one standard which is easy for model caomputation
Standard_scaler=StandardScaler()
#Feature Engineering
#copy the scaled data to different variable
df=train_data.copy()
#placing the columns
df.columns=train_data.columns
#let us take a top 5 variables in unique variable count list count is less
features_unique=['var_68','var_108','var_12','var_91','var_103']
#Creating the Label count encoding
def count_encoder(df,features_unique):
    X_=pd.DataFrame()
    for i in features_unique:
        unique_values=df[i].value_counts()
        value_count_list=unique_values.index.tolist()
        categorical_values=list(range(len(unique_values)))
        label_count=dict(zip(value_count_list,categorical_values))
        X_[i]=df[i].map(label_count)
    X_=X_.add_suffix('_label_encoding')
    df=pd.concat([df,X_],axis=1)    
    return df
#imbalanced data with feature engineering
df_encoded_values=count_encoder(df,features_unique)
imbalanced_data_X_fea_eng,imbalanced_data_Y_fea_eng=SMOTE(random_state=42).fit_resample(df_encoded_values.iloc[:,1:],df_encoded_values.iloc[:,0])
imbalanced_data_X_fea_eng=pd.DataFrame(imbalanced_data_X_fea_eng)
imbalanced_data_Y_fea_eng=pd.DataFrame(imbalanced_data_Y_fea_eng)
imbalanced_data=pd.concat([imbalanced_data_Y_fea_eng,imbalanced_data_X_fea_eng],axis=1)
imbalanced_data.columns=list(df_encoded_values.columns)
imbalanced_data['mean']=imbalanced_data.iloc[:,1:201].mean(axis=1)
imbalanced_data['sum']=imbalanced_data.iloc[:,1:201].sum(axis=1)
imbalanced_data['max']=imbalanced_data.iloc[:,1:201].max(axis=1)
imbalanced_data['min']=imbalanced_data.iloc[:,1:201].min(axis=1)
imbalanced_data['std']=imbalanced_data.iloc[:,1:201].std(axis=1)
imbalanced_data['median']=imbalanced_data.iloc[:,1:201].median(axis=1)
imbalanced_data['skew']=imbalanced_data.iloc[:,1:201].skew(axis=1)
imbalanced_data['kurtosis']=imbalanced_data.iloc[:,1:201].kurtosis(axis=1)
#applying standard scaler on data for better modelling
scaled_data_feature_eng_im=pd.DataFrame(Standard_scaler.fit_transform(imbalanced_data.iloc[:,1:].values))
scaled_data_feature_eng_im=pd.concat([imbalanced_data.iloc[:,0],scaled_data_feature_eng_im],axis=1)
scaled_data_feature_eng_im.columns=list(imbalanced_data.columns)
#data spliting
Data=scaled_data_feature_eng_im.iloc[:,1:]
Data.columns=list(scaled_data_feature_eng_im.columns)[1:]
Target=scaled_data_feature_eng_im.iloc[:,0]
Target.columns=['target']
#spliting the mode with train_test_split where stratify option help class distribution
X_train,X_test,Y_train,Y_test=train_test_split(Data,Target,random_state=0,test_size=0.2,stratify=Target)
#function for comparing model
def validation(model,X,Y,X_test,Y_test):
    model.fit(X,Y)
    predictions=model.predict(X_test)
    print(classification_report(Y_test,predictions),roc_auc_score(Y_test,predictions))
#Logistic regression
validation(LogisticRegression(n_jobs=-1),X_train,Y_train,X_test,Y_test)#auc score 0.86
#KNN
validation(KNN(n_neighbors=5,n_jobs=-1),X_train,Y_train,X_test,Y_test)#0.5
#naive_bayes
validation(GaussianNB(),X_train,Y_train,X_test,Y_test)#auc score:0.85
#random forest
validation(RandomForestClassifier(n_jobs=-1),X_train,Y_train,X_test,Y_test)#auc score 0.90
#xgboost
validation(XGBClassifier(n_jobs=-1),X_train,Y_train,X_test,Y_test)#auc score 0.80

#feature selection method
Random_forest=RandomForestClassifier(n_jobs=-1)
feat_selector=BorutaPy(Random_forest,n_estimators='auto', verbose=2)
feat_selector.fit(Data.values,Target.values)
#check supported features
print(feat_selector.support_)
 
# check ranking of features
print(feat_selector.ranking_)
#where getting the data from suppported features
Data_filtered=feat_selector.transform(Data.values)
Data_filtered=pd.DataFrame(Data_filtered)
X_train,X_test,Y_train,Y_test=train_test_split(Data_filtered,Target,random_state=0,test_size=0.2,stratify=Target)


#hyperparameter tuning
param_grid = {
    "max_depth": (80,110),
    "max_features": (3,11),
    "min_samples_leaf": (3,9),
    "min_samples_split": (8,14),
    "n_estimators": (100,1000)
    }

#above method is taking so much of time so we shifting to bayesian optimization
def objective_function(max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators):
    model=RandomForestClassifier(n_jobs=-1,max_depth=int(max_depth),n_estimators=int(n_estimators),min_samples_split =int(min_samples_split) ,min_samples_leaf=int(min_samples_leaf), max_features= int(max_features))
    model.fit(X_train,Y_train)
    predictions=model.predict(X_test)
    return roc_auc_score(Y_test,predictions)

Bayesian_opt=BayesianOptimization(objective_function,param_grid)
Bayesian_opt.maximize(n_iter=7,init_points=5)
print(Bayesian_opt.max['target'])
print(Bayesian_opt.max['params'])
#Checking the score and setting the best parameter from baysian optimization algorithm.
Random_forest=RandomForestClassifier(n_jobs=-1,max_depth=92,max_features=8,min_samples_leaf=4,min_samples_split=10,n_estimators=183)
Random_forest.fit(X_train,Y_train)
predictions=Random_forest.predict(X_test)
print(classification_report(Y_test,predictions),roc_auc_score(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
confusion_matrix=pd.crosstab(Y_test,predictions)
print(confusion_matrix)
#predicting the test results
test_data_after=count_encoder(test_data,features_unique)
test_data_after['mean']=test_data.iloc[:,0:200].mean(axis=1)
test_data_after['sum']=test_data.iloc[:,0:200].sum(axis=1)
test_data_after['max']=test_data.iloc[:,0:200].max(axis=1)
test_data_after['min']=test_data.iloc[:,0:200].min(axis=1)
test_data_after['std']=test_data.iloc[:,0:200].std(axis=1)
test_data_after['median']=test_data.iloc[:,0:200].median(axis=1)
test_data_after['skew']=test_data.iloc[:,0:200].skew(axis=1)
test_data_after['kurtosis']=test_data.iloc[:,0:200].kurtosis(axis=1)
test_data_filtered=pd.DataFrame(Standard_scaler.fit_transform(test_data_after.iloc[:,0:].values))
test_data_filtered.columns=list(test_data_after.columns)
test_data_filtered=feat_selector.transform(test_data_filtered.values)
test_data_filtered=pd.DataFrame(test_data_filtered)
predictions=pd.DataFrame(Random_forest.predict(test_data_filtered),columns=['target'])
predictions=pd.concat([ID_code,predictions],axis=1)
predictions.to_csv('test_submissions.csv',index=False)
#saving model to joblib files
joblib.dump(feat_selector,'feature_importance.joblib')
joblib.dump(Random_forest,'model.joblib')

