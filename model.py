# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:53:46 2020

@author: team3
"""
import warnings
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
import _pickle as cPickle
warnings.filterwarnings('ignore')
def remove_outliers(data_in):
    '''remove outliers from the dataset by passing all the continuous data c'''
    for col in data_in:
        if data_in[col].dtype in ['float','int'] and col not in ['class']:
            data_in = remove(data_in, col)
    return data_in
def remove(data_in, col):
    '''removes class wise outliers from a column'''
    ds_1 = data_in.loc[data_in['class'] == "hypothyroid"]
    ds_2 = data_in.loc[data_in['class'] == "negative"]
    minim1 = ds_1[col].quantile(0.25)
    maxim1 = ds_1[col].quantile(0.75)
    mid1 = maxim1-minim1
    low1 = minim1-mid1*1.5
    high1 = maxim1+mid1*1.5
    d1 = ds_1.loc[(low1 < ds_1[col]) & (ds_1[col] < high1)]
    minim1 = ds_2[col].quantile(0.25)
    maxim1 = ds_2[col].quantile(0.75)
    mid1 = maxim1-minim1
    low1 = minim1-mid1*1.5
    high1 = maxim1+mid1*1.5
    d2 = ds_2.loc[(low1 < ds_2[col]) & (ds_2[col] < high1)]
    return d1.append(d2)
def train_test_split_smote(data_in):
    '''To split the data into train and test sets and apply smote'''
    print(data_in.dtypes)
    x = data_in.iloc[:, data_in.columns != 'class']
    y = data_in.iloc[:, data_in.columns == 'class']
    osam = SMOTE(random_state=0)
    os_data_x, os_data_y = osam.fit_sample(x, y)
    columns = x.columns
    x_train, x_test, y_train, y_test = train_test_split(os_data_x, os_data_y, train_size=0.80)
    pd.DataFrame(data=os_data_x, columns=columns)
    pd.DataFrame(data=os_data_y, columns=['class'])
    return  x_train, x_test, y_train, y_test  
def rfc_apply(x_train, x_test, y_train, y_test):
    '''To learn model from model with Random Forest Classifier to the data'''
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    predicted = rfc.predict(x_test)
    print("Accuracy :",end='')
    print(accuracy_score(y_test, predicted))
    return rfc
def preprocess(data_in):
    if 'sex' in data_in:
        data_in['sex'] = data_in['sex'].replace('?', 'F')
    data_in = data_in.replace('?', 0)
    if 'age' in data_in:
        data_in['age'] = data_in['age'].astype(np.int)
        data_in['age'] = data_in['age'].replace(0, data_in['age'].median())
    if 'TSH' in data_in:
        data_in['TSH'] = data_in['TSH'].astype(np.float)
    if 'T3' in data_in:
        data_in['T3'] = data_in['T3'].astype(np.float)
    if 'TT4' in data_in:
        data_in['TT4'] = data_in['TT4'].astype(np.float)
    if 'T4U' in data_in:
        data_in['T4U'] = data_in['T4U'].astype(np.float)
    if 'FTI' in data_in:
        data_in['FTI'] = data_in['FTI'].astype(np.float)
    if 'TBG' in data_in:
        data_in.drop('TBG',axis=1)
    data_in = remove_outliers(data_in)
    data_in=change(data_in)
    data_in=RFE(data_in)
    x_train, x_test, y_train, y_test = train_test_split_smote(rmean(data_in))  
    rfc=rfc_apply( x_train, x_test, y_train, y_test )
    return rfc
def change(datasetfs, start=None):
    '''transforms the dataset to be suitable for the algorithms i.e, maps categorical to numerical'''
    datasetfs = datasetfs.replace('F', 1)
    datasetfs = datasetfs.replace('M', 0)
    datasetfs = datasetfs.replace('f', 0)
    datasetfs = datasetfs.replace('t', 1)
    datasetfs = datasetfs.replace('n', 0)
    datasetfs = datasetfs.replace('y', 1)
    datasetfs = datasetfs.replace('negative', 0)
    datasetfs = datasetfs.replace('hypothyroid', 1)
    return datasetfs
def RFE(data_in):
    x = data_in.iloc[:, data_in.columns != 'class']
    y = data_in.iloc[:, data_in.columns == 'class']
    rfc = RandomForestClassifier(random_state=101)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(x, y)        
    x.drop(x.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
    l=['class']
    for i in x:
        l.append(i)
    return data_in[l]
def rmean(data_in):
    '''To perform null value replacement by mean'''
    ds_1 = data_in.loc[(data_in['class'] == "hypothyroid")  |  (data_in['class'] == 1) ]
    ds_2 = data_in.loc[(data_in['class'] == "negative")  | ( data_in['class'] == 0)]
    for col in ds_1:
        if ds_1[col].dtype in ['float']:
            ds_1[col] = ds_1[col].replace(0, ds_1[col].mean())
    for col in ds_2:
        if ds_2[col].dtype in ['float']:
            ds_2[col] = ds_2[col].replace(0, ds_2[col].mean())
    ds_mean = ds_1.append(ds_2)
    return ds_mean
names = ['class', 'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication']
names.extend(['thyroid_surgery', 'query_hypothyroid', 'query_hyperthyroid', 'pregnant', 'sick'])
names.extend(['tumor', 'lithium', 'goitre', 'TSH_measured', 'TSH', 'T3_measured'])
names.extend(['T3', 'TT4_measured', 'TT4', 'T4U_measured', 'T4U', 'FTI_measured'])
names.extend(['FTI', 'TBG_measured', 'TBG'])
dataset = pd.read_csv('Hypothyroid.csv', names=names, header=None)
dataset=dataset.drop('FTI',axis=1)
dataset=dataset.drop('T4U',axis=1)
obj=preprocess(dataset)
cPickle.dump(obj,open('model.pkl','wb'))