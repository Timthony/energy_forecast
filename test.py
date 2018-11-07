# coding=utf-8


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# File system manangement
import os
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics


# In[2]:


app_train = pd.read_csv('public.train.csv')  # 读取训练数据
app_test = pd.read_csv('public.test.csv')  # 读取测试数据

train_id = app_train[['ID']]  # 获取训练id
test_id=app_test[['ID']]  # 获取测试id

app_train_test = [app_train, app_test]
app_train_test = pd.concat(app_train_test)

app_train_test=app_train_test.mask(app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))
app_train_test=app_train_test.fillna(method='ffill')

app_train_test.to_csv('data_prc.csv',index=False)


app_train= train_id.merge(app_train_test, on='ID', how='left')
app_test= test_id.merge(app_train_test, on='ID', how='left')
app_test=app_test.drop(columns='发电量')

app_train.to_csv('train_prc.csv',index = False)
app_test.to_csv('test_prc.csv',index = False)
