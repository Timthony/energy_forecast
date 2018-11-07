# coding=utf-8
# In[7]:


# %load my_preprocessing.py

import numpy as np
#import load_data
import pandas as pd

def data_corection(data):

    ID=data['ID']

    for feature_num, feature in enumerate(data.keys()):

        print('正在处理的特征：',feature)

        Q1 = np.percentile(data[feature],25)
        Q3 = np.percentile(data[feature],75)
        step = (Q3-Q1)
        if feature in ['平均功率','功率A','功率B','功率C','former_power','peak_value','I_eta']:
            step=step*1.5
        elif feature in ['电压A','电压B','电压C']:
            step=step*100
        elif feature in ['光照强度','现场温度','板温']:
            step=step*1.5
        elif feature in ['电流A','电流B','电流C']:
            step=step*3
        elif feature in ['转换效率A','转换效率B','转换效率C','转换效率']:
            step=step*10
        elif feature in ['风向']:
            step=step*1.4
        elif feature in ['风速']:
            step=step*4
        else:
            step=step*1000

        feature_index=data[~((data[feature] >= (Q1 - step)) & (data[feature] <= (Q3 + step)))].index

        for i in range(len(feature_index)):
            if feature_index[i]==0:
                j=feature_index[i]+1
                while j in feature_index:
                    j=j+1
                data.iloc[feature_index[i],feature_num]=data.iloc[j,feature_num]
            else:
                j=feature_index[i]-1
                while j in feature_index:
                    j=j-1
                data.iloc[feature_index[i],feature_num]=data.iloc[j,feature_num]
    return data

def scale(features,scaler):

    scaler.fit(features)
    scaled_features=scaler.transform(features)
    return scaled_features,scaler

if __name__=='__main__':

    print('加载数据...')
    train_data=pd.read_csv('public.train.csv')

    print(['原始训练数据：',train_data.shape])
    test_data=pd.read_csv('public.test.csv')
    print(['测试数据：',test_data.shape])

    data=test_data.append(train_data.drop('发电量',axis=1))
    data.sort_values('ID',inplace=True)
    data=data.reset_index(drop=True)
    print('合并后的数据：',data.shape)

    data_prc=data_corection(data.copy())

    data_prc.to_csv('data_prc_lstm.csv',index=None)
    print('数据预处理完毕')