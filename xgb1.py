# coding=utf-8
# In[4]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
app_train = pd.read_csv('train_all_features2.csv')
print('Training data shape: ', app_train.shape)
app_test = pd.read_csv('test_all_features2.csv')
print('Testing data shape: ', app_test.shape)
train_id = app_train[['ID']]
test_id = app_test[['ID']]
app_train_test = [app_train, app_test]
app_train_test = pd.concat(app_train_test)
app_train_test = app_train_test.mask(app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))
'''may change ffill'''
app_train_test = app_train_test.fillna(method='ffill')
app_train = train_id.merge(app_train_test, on='ID', how='left')
app_test = test_id.merge(app_train_test, on='ID', how='left')
app_test = app_test.drop(columns='发电量')
app_train = app_train.drop(columns='peak_value')
app_test = app_test.drop(columns='peak_value')

'''a'''
app_train['理论输出'] = app_train['光照强度'] * app_train['转换效率']
app_test['理论输出'] = app_test['光照强度'] * app_test['转换效率']
'''b'''
app_train['温差'] = app_train['板温'] - app_train['现场温度']
app_test['温差'] = app_test['板温'] - app_test['现场温度']
'''c'''
app_train['实际功率'] = app_train['转换效率'] * app_train['平均功率']
app_test['实际功率'] = app_test['转换效率'] * app_test['平均功率']
'''d'''
# app_train['风力X风向']=app_train['风向']*app_train['风速']
# app_test['风力X风向']=app_test['风向']*app_test['风速']

app_train['实际温度'] = app_train['转换效率'] * app_train['现场温度']
app_test['实际温度'] = app_test['转换效率'] * app_test['现场温度']
'''开始瞎jb蒙'''
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

# app_train['C_4']=app_train['功率A']/app_train['风速']
# app_test['C_4']=app_test['功率A']/app_test['风速']

# app_train['C_5']=app_train['功率B']/app_train['风速']
# app_test['C_5']=app_test['功率B']/app_test['风速']
#
# app_train['C_6']=app_train['C_4']*app_train['C_4']
# app_test['C_6']=app_test['C_4']*app_test['C_4']
# app_train['C']=app_train['电流B']-app_train['电流C']
# app_test['C']=app_test['电流B']-app_test['电流C']

# app_train['C_3']=app_train['dis2peak']*app_train['平均功率']
# app_test['C_3']=app_test['dis2peak']*app_test['平均功率']

poly_features = app_train[['板温', '现场温度', '光照强度', '风速', '风向']]
poly_features_test = app_test[['板温', '现场温度', '光照强度', '风速', '风向']]

# imputer for handling missing values
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')

poly_target = app_train['发电量']

# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

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
#################################################zth###################################



from sklearn.model_selection import KFold
import gc


def my_scorer(y_true, y_predicted, X_test):
    loss_train = np.sum((y_true - y_predicted) ** 2, axis=0) / (X_test.shape[0])  # RMSE
    loss_train = loss_train ** 0.5
    score = 1 / (1 + loss_train)
    return score


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
    # valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # Create the model
        model = xgb.XGBRegressor(objective='reg:linear', n_estimators=16000, min_child_weight=1, num_leaves=20,
                                 learning_rate=0.01, max_depth=6, n_jobs=20,
                                 subsample=0.6, colsample_bytree=0.4, colsample_bylevel=1)

        # Train the model
        model.fit(train_features, train_labels,
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  early_stopping_rounds=300, verbose=600)

        # Record the best iteration
        best_iteration = 16000

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict(test_features) / k_fold.n_splits
        train_predictions += model.predict(features) / k_fold.n_splits
        # Record the out of fold predictions
        out_of_fold = model.predict(valid_features) / k_fold.n_splits

        # Record the best score
        train_score = my_scorer(valid_labels, out_of_fold, valid_features)

        # valid_scores.append(valid_score)
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
xgb2lgb_train = pd.read_csv('train_all_features2.csv')
xgb2lgb_test = pd.read_csv('test_all_features2.csv')
xgb2lgb_train['xgb发电量'] = train_sub['发电量']
xgb2lgb_test['xgb发电量'] = submission['发电量']
xgb2lgb_train.to_csv('xgb2lgb_train.csv', index=False)
xgb2lgb_test.to_csv('xgb2lgb_test.csv', index=False)  ##  0.080844   0.84669334000
##  0.079003    0.84672240000
# 0.093215     0.8457
# 0.078988   0.84892
# 0.078988   0.84897390000
# 0.84902996000
# 0.066291  0.84906185000
# 0.065589  0.84901553000
# 0.068688  0.84901680000
# 0.166649  0.84985
