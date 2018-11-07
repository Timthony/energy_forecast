# coding=utf-8

# In[6]:


import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os
import numpy as np
import pandas as pd
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics

app_train = pd.read_csv('xgb2lgb_train.csv')
print('Training data shape: ', app_train.shape)
app_test = pd.read_csv('xgb2lgb_test.csv')
print('Testing data shape: ', app_test.shape)

train_id = app_train[['ID']]
test_id = app_test[['ID']]
app_train_test = [app_train, app_test]
app_train_test = pd.concat(app_train_test)
app_train_test = app_train_test.mask(app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))

app_train_test = app_train_test.fillna(method='ffill')
app_train = train_id.merge(app_train_test, on='ID', how='left')
app_test = test_id.merge(app_train_test, on='ID', how='left')
app_test = app_test.drop(columns='发电量')
app_train = app_train.drop(['peak_value'], axis=1)
app_test = app_test.drop(['peak_value'], axis=1)

app_train['理论输出'] = app_train['光照强度'] * app_train['转换效率']
app_test['理论输出'] = app_test['光照强度'] * app_test['转换效率']

app_train['温差'] = app_train['板温'] - app_train['现场温度']
app_test['温差'] = app_test['板温'] - app_test['现场温度']

app_train['实际功率'] = app_train['转换效率'] * app_train['平均功率']
app_test['实际功率'] = app_test['转换效率'] * app_test['平均功率']

# app_train['风力X风向']=app_train['风向']*app_train['风速']
# app_test['风力X风向']=app_test['风向']*app_test['风速']

app_train['实际温度'] = app_train['转换效率'] * app_train['现场温度']
app_test['实际温度'] = app_test['转换效率'] * app_test['现场温度']

app_train['电压差A'] = app_train['电流A'] - app_train['电流B']
app_test['电压差A'] = app_test['电流A'] - app_test['电流B']

app_train['cde'] = app_train['电压A'] / app_train['转换效率A']
app_test['cde'] = app_test['电压A'] / app_test['转换效率A']
app_train['cde1'] = app_train['电压B'] / app_train['转换效率B']
app_test['cde1'] = app_test['电压B'] / app_test['转换效率B']
app_train['cde2'] = app_train['电压C'] / app_train['转换效率C']
app_test['cde2'] = app_test['电压C'] / app_test['转换效率C']

# app_train['abk']=app_train['abk']*app_train['abk']
# app_test['abk']=app_test['abk']*app_test['abk']
# app_train['iuo']=app_train['光照强度']*np.cos((app_train['ID']))
# app_test['iuo']=app_test['光照强度']*np.cos((app_test['ID']))
app_train['cdex'] = app_train['cde'] * app_train['cde']
app_test['cdex'] = app_test['cde'] * app_test['cde']
app_train['cdex1'] = app_train['cde1'] * app_train['cde1']
app_test['cdex1'] = app_test['cde1'] * app_test['cde1']
app_train['cdex2'] = app_train['cde2'] * app_train['cde2']
app_test['cdex2'] = app_test['cde2'] * app_test['cde2']

# print(app_train['dis2peak'])
app_train['C_1'] = app_train['dis2peak'] * app_train['dis2peak']
app_test['C_1'] = app_test['dis2peak'] * app_test['dis2peak']

app_train['C_2'] = app_train['dis2peak'] * app_train['光照强度']
app_test['C_2'] = app_test['dis2peak'] * app_test['光照强度']

app_train['C_3'] = app_train['电流B'] - app_train['电流C']
app_test['C_3'] = app_test['电流B'] - app_test['电流C']

app_train['C_4'] = app_train['功率A'] / app_train['风速']
app_test['C_4'] = app_test['功率A'] / app_test['风速']

app_train['C_5'] = app_train['功率B'] / app_train['风速']
app_test['C_5'] = app_test['功率B'] / app_test['风速']

app_train['C_6'] = app_train['C_4'] * app_train['C_4']
app_test['C_6'] = app_test['C_4'] * app_test['C_4']

app_train['C_7'] = app_train['电压A'] / app_train['风速']
app_test['C_7'] = app_test['电压A'] / app_test['风速']

app_train['C_8'] = app_train['电流A'] / app_train['风速']
app_test['C_8'] = app_test['电流A'] / app_test['风速']
#
app_train['C_9'] = app_train['风向'] * app_train['转换效率A']
app_test['C_9'] = app_test['风向'] * app_test['转换效率A']
#
# app_train['C_11']=app_train['板温']/app_train['现场温度']
# app_test['C_11']=app_test['板温']/app_test['现场温度']
#
# app_train['C_10']=app_train['功率A']*app_train['光照强度']
# app_test['C_10']=app_test['功率A']*app_test['光照强度']

poly_features = app_train[['板温', '现场温度', '光照强度', '风速', '风向']]
poly_features_test = app_test[['板温', '现场温度', '光照强度', '风速', '风向']]

# imputer for handling missing values
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')

poly_target = app_train['发电量']

# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree=2)

poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)

poly_features = pd.DataFrame(poly_features,
                             columns=poly_transformer.get_feature_names(['板温', '现场温度', '光照强度', '风速', '风向']))

# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
# print(poly_corrs)
# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test,
                                  columns=poly_transformer.get_feature_names(['板温', '现场温度', '光照强度', '风速', '风向']))
''''''

''''''
# Merge polynomial features into training dataframe
poly_features['ID'] = app_train['ID']
app_train_poly = app_train.merge(poly_features, on='ID', how='left')

# Merge polnomial features into testing dataframe
poly_features_test['ID'] = app_test['ID']
app_test_poly = app_test.merge(poly_features_test, on='ID', how='left')

# Align the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

app_train_poly['发电量'] = poly_target
app_train = app_train_poly
app_test = app_test_poly

# app_train.to_csv('xgb_poly_timetrain.csv',index=False)
# app_test.to_csv('xgb_poly_timetest.csv',index=False)

print('Training data with polynomial features shape: ', poly_features.shape)
print('Testing data with polynomial features shape:  ', poly_features_test.shape)
print('Training data with polynomial features shape: ', app_train.shape)
print('Testing data with polynomial features shape:  ', app_test.shape)

app_train['ID'] = train_id
app_test['ID'] = test_id

from sklearn.model_selection import KFold
import gc


def model(features, test_features, encoding='ohe', n_folds=4):
    # Extract the ids
    train_ids = features['ID']
    test_ids = test_features['ID']

    # Extract the labels for training
    labels = features['发电量']

    # Remove the ids and target
    features = features.drop(columns=['ID', '发电量'])
    test_features = test_features.drop(columns=['ID'])

    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)

        test_features = pd.get_dummies(test_features)

        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join='inner', axis=1)

        # No categorical indices to record
        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        # Create a label encoder
        label_encoder = LabelEncoder()

        # List for storing categorical indices
        cat_indices = []

        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)

    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    train_predictions = np.zeros(features.shape[0])
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # Create the model
        model = lgb.LGBMRegressor(objective='regression', n_estimators=12000, min_child_samples=20, num_leaves=26,
                                  learning_rate=0.005, feature_fraction=0.6,
                                  subsample=0.4, n_jobs=-1, random_state=50)

        # Train the model
        model.fit(train_features, train_labels, eval_metric='rmse',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], categorical_feature=cat_indices,
                  early_stopping_rounds=1000, verbose=600)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict(test_features, num_iteration=best_iteration) / k_fold.n_splits
        train_predictions += model.predict(features, num_iteration=best_iteration) / k_fold.n_splits
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict(valid_features, num_iteration=best_iteration) / k_fold.n_splits

        # Record the best score
        valid_score = model.best_score_['valid']['rmse']
        train_score = model.best_score_['train']['rmse']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'ID': test_ids, '发电量': test_predictions})
    train_sub = pd.DataFrame({'ID': train_ids, '发电量': train_predictions})
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    # valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    # valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metric = pd.DataFrame({'fold': fold_names,
                           'train': train_scores,
                           })

    return submission, feature_importances, metric, train_sub


submission, fi, metric, train_sub = model(app_train, app_test)
print('Baseline metrics')
print(metric)
submission.to_csv('lgb_final_result.csv', index=False)  ##  0.080844   0.84669334000

# 0.012138    0.85068184000
# 0.012227    0.85057590000
# 0.009218     0.85061880000
# 0.010561    0.85058580000
# 0.011587     0.85059136000
# 0.011767      850668
# 0.011500     0.85065
# 0.012576    0.85063404000
# 0.011863  850617
# 0.012066    0.85058500000
# 0.012996    0.85058284000
# 0.0110560.85061100000
