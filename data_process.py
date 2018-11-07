# coding=utf-8

# In[8]:


import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import PolynomialFeatures

def add_former_power(features):

    mean_power=features['平均功率']
    former_power=np.zeros(features.shape[0])
    for i in range(1,features.shape[0]):
        former_power[i]=mean_power[i-1]
    former_power[0]=mean_power[0]

    features.insert(1,'former_power',former_power)
    # features.to_csv('data/features_former_power.csv',index=None)

    return features

def add_dis2peak(data):
    peaks_IDs_unchecked = [90, 309, 466, 686, 844, 1046, 1209, 1402, 1597, 1775, 1976, 2163, 2349, 2540,
             2723, 2937, 3104, 3263, 3440, 3646, 3844, 3985, 4178, 4357, 4547, 4745, 4915,
             5175, 5387, 5607, 5803, 6018, 6229, 6443, 6635, 6816, 7017, 7206, 7432, 7633,
             7846, 8018, 8226, 8396, 8589, 8806, 8998, 9169, 9390, 9631, 9807, 10018, 10250,
             10432, 10645, 10825, 10998, 11230, 11428, 11632, 11850, 12054, 12264, 12476,
             12689, 12904, 13102, 13312, 13544, 13708, 13915, 14125, 14317, 14555, 14759,
             14952, 15173, 15399, 15612, 15813, 16036, 16240, 16436, 16651, 16838, 17028,
             17223, 17399, 17645, 17840]

    peak_IDs = []
    peak_index=[]
    ID=data['ID']
    for ID_i in peaks_IDs_unchecked:
        for j, ID_j in enumerate(ID.values):
            if ID_i >= ID_j and ID_i < ID[j+1]:
                peak_IDs.append(ID_j)
                peak_index.append(j)


    dis2peak = []
    peak_value=[]
    mean_power=data['平均功率']
    for id in data['ID']:
        mindis = np.abs(id - peak_IDs[0])
        peak_row=peak_index[0]
        for i,peak_id in enumerate(peak_IDs):
            if np.abs(id-peak_id)<mindis:
                mindis=np.abs(id-peak_id)
                peak_row=peak_index[i]
        dis2peak.append(mindis)
        peak_value.append(mean_power[peak_row])

    data.insert(1, 'dis2peak', dis2peak )
    data.insert(1, 'peak_value', peak_value )
    return data

def add_power_mean_std(data):

    mean_power=[]
    std_power=[]

    for dis in enumerate(data['dis2peak']):
        mean_power_i=np.mean(data[data['dis2peak']==dis[1]]['平均功率'])
        mean_power.append(mean_power_i)
        std_power_i=np.std(data[data['dis2peak']==dis[1]]['平均功率'])
        std_power.append(std_power_i)

    data.insert(17,'mean_power',mean_power)
    data.insert(17,'std_power',std_power)

    return data

def add_mean_board_temperature(data,T=210):

    board_temperature=data['板温']
    mean_board_temperature=[]

    for i in range(len(board_temperature)):
        temperature_i=[]
        if i<len(board_temperature)-T:
            temperature_i=np.sum(board_temperature[i:i+T])*1.0/T
        else:
            temperature_i=np.sum(board_temperature[i-T:i])*1.0/T
        mean_board_temperature.append(temperature_i)
    data.insert(1,'mean_board_temp',mean_board_temperature)

    return data

def add_wind(data,T=20):

    wind0=list(data['风速']*data['风向'])
    wind=[]
    for i in range(len(data['风向'])):
        wind_i=[]
        if i<T/2:
            wind_i=np.sum(wind0[i:i+T])*1.0/T
        elif i<len(data['风向'])-T/2:
            wind_i=np.sum(wind0[i-int(T/2):i+int(T/2)])*1.0/T
        else:
            wind_i=np.sum(wind0[i-T:i])*1.0/T
        wind.append(wind_i)
    data.insert(17,'wind',wind0)

    return data

def add_I_eta(data):

    data.insert(6,'I_eta',data['光照强度']*data['转换效率'])

    return data

def add_P_eta(data):
    data.insert(16,'P_eta',data['转换效率']*data['平均功率'])
    return data

def add_temp_diff(data):
    data.insert(10,'temp_diff',data['板温']-data['现场温度'])
    return data

def add_idc(data):
    data.insert(11,'idc_A',data['电流A']/(data['转换效率A']+0.001))
    data.insert(11,'idc_B',data['电流B']/(data['转换效率B']+0.001))
    data.insert(11,'idc_C',data['电流C']/(data['转换效率C']+0.001))
    return data

def add_vdc(data):
    data.insert(11,'vdc_A',data['电压A']/(data['转换效率A']+0.001))
    data.insert(11,'vdc_B',data['电压B']/(data['转换效率B']+0.001))
    data.insert(11,'vdc_C',data['电压C']/(data['转换效率C']+0.001))
    return data

def add_temp_diff_light(data):
    data.insert(10,'温差乘以光强',np.abs((data['板温']-data['现场温度'])*data['光照强度']))
    return data

def add_PN_I(data):
    I0=4.215*10**(-14)*np.exp(0.1539*data['板温'])
    q=1.6*10**(-9)
    K=0.86*10**(-4)
    U=data['电压A']
    T=data['板温']+273.15
    Id=I0*(np.exp((q*U)/(K*T))-1)
    data.insert(13,'PN_I',Id*10**16.5 )
    return data

def add_vdc_square(data):
    data.insert(14,'vdc_A_square',(data['电压A']/(data['转换效率A']+0.1))**2)
    data.insert(14,'vdc_B_square',(data['电压B']/(data['转换效率B']+0.1))**2)
    data.insert(14,'vdc_C_square',(data['电压C']/(data['转换效率C']+0.1))**2)
    return data

def add_idc_square(data):
    data.insert(17,'idc_A_square',(data['电流A']/(data['转换效率A']+0.1))**2)
    data.insert(17,'idc_B_square',(data['电流B']/(data['转换效率B']+0.1))**2)
    data.insert(17,'idc_C_square',(data['电流C']/(data['转换效率C']+0.1))**2)
    return data

def add_power_divide_speed(data):
    data.insert(14,'power_divide_speed_A',data['功率A']/(data['风速']+0.001))
    data.insert(14,'power_divide_speed_B',data['功率B']/(data['风速']+0.001))
    data.insert(14,'power_divide_speed_C',data['功率C']/(data['风速']+0.001))
    data.insert(14,'power_divide_speed_A2',(data['功率A']/(data['风速']+0.001))**2)
    data.insert(14,'Ic_Ib',data['电流B']-data['电流C'])
    return data

def add_poly_features(data,column_names):
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])

    return rest_features

def do_feature_project(features):

    print('添加前一时刻平均功率...')
    features = add_former_power(features)

    print('添加与平均功率峰的值距离和平均功率峰值大小...')
    features=add_dis2peak(features)

    print('添加平均功率的均值和方差...')
    features=add_power_mean_std(features)

    print('添加平均板温...')
    features=add_mean_board_temperature(features)

    print('添加风速和风向的乘积...')
    features=add_wind(features,T=20)

    print('添加光照强度和转化效率的乘积')
    features=add_I_eta(features)

    print('添加功率乘以效率')
    features=add_P_eta(features)

    print('添加温差')
    features=add_temp_diff(features)

    print('添加电流除以转换效率')
    features=add_idc(features)

    print('添加电压除以转换效率，然后平方')
    features=add_vdc(features)

    print('添加电压除以转换效率，然后平方')
    features=add_vdc_square(features)

    print('添加电流除以转换效率，然后平方')
    features=add_idc_square(features)

    print('添加温差乘以光强')
    features=add_temp_diff_light(features)

    print('添加PN结电流')
    features=add_PN_I(features)

    print('添加功率除以风速')
    # features=add_power_divide_speed(features)


    print('添加PolyFeatures')
    # column_names=['板温','光照强度','转换效率A','电压A','电流A','风速','风向','temp_diff']
    column_names=['板温','光照强度','转换效率A','电压A','电流A','风速','风向','temp_diff','平均功率']
    features=add_poly_features(features,column_names)

    # features=features.drop([ '电压A 电流A', '光照强度 转换效率'],axis=1)

    print('正在保存特征...')
    features.to_csv('features_lstm.csv',index=None)

    print('特征构造完毕!总特征数量为：',features.shape[1])

    return features

if __name__ == '__main__':

    t0=datetime.datetime.now()
    print(t0)
    train_data=pd.read_csv('public.train.csv')
    train_ID=train_data['ID']
    test_data=pd.read_csv('public.test.csv')
    test_ID=test_data['ID']

    features=pd.read_csv('data_prc_lstm.csv')
    # print(features.head(10))

    features=do_feature_project(features)
    # features=pd.read_csv('data/features.csv')

    train_features=features[features['ID'].isin(train_ID)]
    train_features.reset_index(drop=True)
    train_features.insert(train_features.shape[1],'发电量',train_data['发电量'].values)
    train_data=train_features

    test_data=features[features['ID'].isin(test_ID)]
    test_data.reset_index(drop=True)

    print('正在保存新的训练集和测试集...')
    train_data.to_csv('train_all_features_lstm.csv',index=None)
    test_data.to_csv('test_all_features_lstm.csv',index=None)
    print('正在保存新的训练集和测试集...')

