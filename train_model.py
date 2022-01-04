# -*- coding: utf-8 -*-
import pickle
import random
import pandas as pd
import numpy as np

from xgboost.sklearn import XGBClassifier # 用於建立RandomizedSearchCV的XGBoost
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV # 用於建立RandomizedSearchCV的XGBoost
from sklearn.model_selection import train_test_split
import xgboost as xgb # 一般的XGBoost用於和RandomizedSearchCV的XGBoost做比較

import matplotlib.pyplot as plt

mood_mapper = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
song_mapper = {0:'pop', 1:'soft', 2:'funny', 3:'jazz', 4:'lofi'}

## Load txt data
file_object = open(r"picast_data.txt","r")
row = []
 
for line in file_object.read().splitlines():
  row.append(line.split(' '))

df = pd.DataFrame(row, columns=['temperature', 'humidity', 'emotion', 'genre'])
df = df.dropna()


## Build Recommendation Model"""

# print出XGBoost的feature importance
def print_xgb_feature_importance(feature_df, xgb_model):

  plt.barh(feature_df.columns, xgb_model.feature_importances_)
  plt.title('XGBoost feature impotance')
  plt.xlabel('importance')
  plt.ylabel('feature')
  plt.show()

input_col = ['temperature','humidity','emotion']
output_col = ["genre"]

X = df[input_col].values
y = df[output_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

########################### build XGBoost with RandomSearchCV ###########################
# Souce : https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
# Source2 : https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

xgb_base = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', silent=True, nthread=1) 

# A parameter grid for XGBoost
params = {
      'min_child_weight': [1, 5, 10], # Minimum sum of instance weight (hessian) needed in a child.
      'gamma': [0, 0.1, 0.3, 0.5, 1, 1.5, 2, 5], # Minimum loss reduction required to make a further partition on a leaf node of the tree.
      'subsample': [0.6, 0.8, 1.0], # Subsample ratio of the training instances
      'colsample_bytree': [0.6, 0.8, 1.0], # subsample ratio of columns when constructing each tree
      'scale_pos_weight': [0.5, 0.6, 0.7, 1], # Control the balance of positive and negative weights
      'max_depth': range(3,10,2)
      }
folds, param_comb = 3, 5
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
random_search = RandomizedSearchCV(xgb_base, 
                  param_distributions=params, 
                  n_iter=param_comb, 
                  scoring='roc_auc', 
                  n_jobs=4, 
                  cv=skf.split(X_train,y_train), 
                  verbose=3, 
                  random_state=1001 )
# Here we go
random_search.fit(X_train, y_train)

################################## build normal XGBoost ##################################
xgb_model = xgb.XGBClassifier(colsample_bytree=1.0, gamma=1, learning_rate=0.02, max_depth=5,
              n_estimators=600, nthread=1, objective='multi:softprob',
              silent=True, subsample=0.6)
xgb_model.fit(X_train, y_train)


####################################### Evaluation #######################################

print_xgb_feature_importance(df[input_col], xgb_model)

result = loaded_model.score(X_test, y_test)
print(result)

 
filename = 'music_recommendation_picast.pkl'
pickle.dump(xgb_model, open(filename, 'wb'))

 


