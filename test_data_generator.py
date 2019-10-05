# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 23:40:46 2019

@author: MUKHESH
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

def test_data_generator(file):
    features_unique=['var_68','var_108','var_12','var_91','var_103']
    test_data=pd.read_csv(file)
    if(test_data.shape[1]<201):
        print('Insuffient variables')
        return -1
    test_data.drop(columns=['ID_code'],inplace=True)
    Standard_scaler=StandardScaler()
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
    return test_data_filtered
 