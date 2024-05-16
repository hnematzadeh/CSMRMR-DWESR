import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_absolute_error

from itertools import product
from sklearn.feature_selection import RFE
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv
import time
from ypstruct import struct
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from numpy.linalg import norm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot as plt


from scipy.cluster.hierarchy import linkage, fcluster
import random
import statsmodels.api as sm
from time import time
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")



###  RandomForest Approach can not improve  mae
from time import time
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
t0=time()
# Train a Random Forest model and calculate feature importance scores
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importance_scores = rf.feature_importances_

# Rank the features based on their importance scores
ranked_features = np.argsort(importance_scores)[::-1]

# Select the top K features
K = 1
selected_features = X.columns[ranked_features[:K]]
t1=time()
# Print the selected features
print('Selected Features:', selected_features)
print(t1-t0)
E=selected_features


######## Correlation coefficinet linear regression  
from time import time 
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
t0=time()
regressor = LinearRegression()
regressor.fit(X, y)

# Define the target variable

# Compute the correlation matrix for all variables
corr_matrix = np.corrcoef(X.values.T)

# Print the correlation matrix
# print("Correlation matrix: \n", corr_matrix)

# Select the top 2 features based on their absolute correlation with the target variable
corr_with_y = corr_matrix[-1, :-1]
feature_ranks = np.argsort(np.abs(corr_with_y))[::-1]
top_features = X.columns[feature_ranks][:1]
t1=time()
# Print the selected top 2 features
print("Top 1 features: ", top_features)
print(t1-t0)
E=top_features

############  linear regression coefficients   (LRC)
from time import time
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
t0=time()
regressor = LinearRegression()
regressor.fit(X, y)

# Define the target variable

# Select the top k features based on f-regression scores
coefficients = regressor.coef_

# Rank the features based on the absolute value of the coefficients
# feature_ranks = np.argsort(np.abs(coefficients))[::-1]
feature_ranks = np.argsort(np.abs(coefficients))[::-1]

# Select the top 2 features based on their coefficients
top_features = X.columns[feature_ranks][:23]
t1=time()
# Print the selected top 2 features
print("Top 1 features: ", top_features)
print(t1-t0)
E=top_features
##### Lasso#################### LassoCV by cross validation finds the best alpha itself
from sklearn.linear_model import LassoCV
reg = LassoCV(cv=10)
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
# fit the model to the data
reg.fit(X, y)

# get the indices of the selected features
# selected_features_idx = np.where(reg.coef_ != 0)[0]

# # get the names of the selected features
# selected_features_names = X.columns[selected_features_idx]

# # get the number of selected features
# num_selected_features = len(selected_features_idx)

# print("Selected Features:", selected_features_names)
# print("Number of Selected Features:", num_selected_features)

# E=selected_features_names



###### Top 1 features

abs_coef = np.abs(reg.coef_)

# get the indices of the sorted coefficients in descending order
sorted_coef_idx = np.argsort(abs_coef)[::-1]

# get the names of the top 2 features
top_1_features = X.columns[sorted_coef_idx[:1]]

print("Top 1 Features:", top_1_features)
E=top_1_features 

##############Lasso without cross validation

from time import time
from sklearn.linear_model import Lasso
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
# Load data
data.index = pd.to_datetime(data['FechaRecepcion'])
data.drop(columns='FechaRecepcion', inplace=True)

# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)

t0=time()
# Create a Lasso object with alpha=0.1
reg = Lasso(alpha=0.1)

# Fit the model to the data
reg.fit(X, y)

# Get the coefficients of the model
coef = reg.coef_

# Get the absolute coefficients
abs_coef = np.abs(coef)

# Get the indices of the sorted coefficients in descending order
sorted_coef_idx = np.argsort(abs_coef)[::-1]

# Get the names of the top 2 features
top_1_features = X.columns[sorted_coef_idx[:1]]
t1=time()
print("Top 1 Features:", top_1_features)
print(t1-t0)
E = top_1_features




######################


from sklearn.linear_model import Lasso
import numpy as np
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index = pd.to_datetime(data['FechaRecepcion'])
data.drop(columns='FechaRecepcion', inplace=True)

# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)

# Create a Lasso object with alpha=0.1
reg = Lasso(alpha=0.1)

# Fit the model to the data
reg.fit(X, y)

# Get the indices of the selected features
selected_features_idx = np.where(reg.coef_ != 0)[0]

# Get the names of the selected features
selected_features_names = X.columns[selected_features_idx]

# Get the number of selected features
num_selected_features = len(selected_features_idx)

print("Selected Features:", selected_features_names)
print("Number of Selected Features:", num_selected_features)

E = selected_features_names
#####################mutual information
from sklearn.feature_selection import mutual_info_regression
from time import time
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
t0=time()
# compute the mutual information scores for each feature
mi_scores = mutual_info_regression(X, y)

# rank the features based on their mutual information scores
feature_ranks = np.argsort(mi_scores)[::-1]

# select the top 2 features based on their mutual information scores
top_features = X.columns[feature_ranks][:1]
t1=time()

# print the selected top 2 features
print("Top 1 features based on mutual information scores: ", top_features)
print(t1-t0)
E=top_features


#################XGboost###############

import xgboost as xgb
from time import time
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
model = xgb.XGBRegressor()
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
# Fit model on entire dataset
t0=time()
model.fit(X, y)

# Get feature importance scores
importance_scores = model.feature_importances_

# Get indices of top 2 features
top_features = importance_scores.argsort()[-1:]
t1=time()
# Extract names of top 2 features
feature_names = X.columns
top_feature_names = feature_names[top_features]

# Print top 2 feature names
print("Top 1 features:", top_feature_names)
print(t1-t0)
E=top_feature_names

################# Decision tree regressor
from time import time
from sklearn.tree import DecisionTreeRegressor
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
t0=time()
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Get feature importance scores and feature names
importance_scores = model.feature_importances_
feature_names = X.columns

# Sort feature importance scores in descending order
sorted_idx = importance_scores.argsort()[::-1]

# Select top 1 features based on importance scores
top_features = feature_names[sorted_idx][:1]
t1=time()
# Print top 1 features
print("Top 1 features:", top_features)
print(t1-t0)
E=top_features

#############Backward SS
###### backward feature selection
from time import time
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
# instantiate model
model3 = RandomForestRegressor(n_estimators=100, random_state=42)
# select features
t0=time.time()
selector = SequentialFeatureSelector(estimator=model3, n_features_to_select=1, direction='forward', cv=None, scoring='r2')
selector.fit_transform(X,y)
# check names of features selected
feature_names = np.array(X.columns)
E=feature_names[selector.get_support()]
t1=time.time()
print("time = ",  t1-t0)
print(E)


#################SFS Backward/forward with no cross validation###############
from time import time
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
# import the mlxtend version of SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
# instantiate model
model3 = RandomForestRegressor(n_estimators=100, random_state=42)
# select features
t0=time.time()
# use the mlxtend version of SequentialFeatureSelector and set cv=0
selector = SequentialFeatureSelector(estimator=model3, k_features=1, forward=False, cv=0, scoring='r2')
selector.fit_transform(X,y)
# check names of features selected
feature_names = np.array(X.columns)
# convert the tuple to a list before indexing
E=feature_names[list(selector.k_feature_idx_)]
t1=time.time()
print("time = ",  t1-t0)
print(E)


############## Correlation coefficient by f_regression (F_statistic)
from time import time
from sklearn.feature_selection import SelectKBest, f_regression
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
t0=time()
# Select the top 2 features
selector = SelectKBest(score_func=f_regression, k=1)
X_top_2 = selector.fit_transform(X, y)

# Get the indices of the selected features
selected_features_idx = selector.get_support(indices=True)

# Get the names of the selected features
selected_features_names = X.columns[selected_features_idx]
t1=time()
print("Selected Features:", selected_features_names)
print(t1-t0)
E=selected_features_names



####### Bagging with Randomforest########
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import pandas as pd
from time import time
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
# Load your dataset into a pandas DataFrame
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)


# Define the base estimator
# base_estimator = LassoCV(cv=10)
t0=time()

base_estimator = DecisionTreeRegressor(random_state=42)

# base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
# Define the bagging regressor as the feature selector
feature_selector = BaggingRegressor(base_estimator=base_estimator, n_estimators=100, random_state=42)

# Fit the feature selector to the data
feature_selector.fit(X, y)

# Get the feature importance scores
importance_scores = feature_selector.estimators_features_

# Sum the feature importance scores across all estimators
importance_sum = sum(importance_scores)
# Get the top 2 feature indices
top_feature_indices = importance_sum.argsort()[::-1][:1]

# Get the names of the selected features
selected_features_names = X.columns[top_feature_indices]
t1=time()
print("Selected Features:", selected_features_names)
print(t1-t0)
E=selected_features_names

############## Adaboost

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from time import time
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
# Load your dataset into a pandas DataFrame
data.index = pd.to_datetime(data['FechaRecepcion'])
data.drop(columns='FechaRecepcion', inplace=True)

# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
t0=time()
# Define the base estimator
base_estimator = DecisionTreeRegressor()

# Create an AdaBoostRegressor with a decision tree as the base estimator
adaboost = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, random_state=42)

# Fit the AdaBoostRegressor
adaboost.fit(X, y)

# Get feature importance scores
importance_scores = adaboost.feature_importances_

# Create a dictionary of feature importance scores and names
importance_dict = {X.columns[i]: importance_scores[i] for i in range(len(X.columns))}

# Sort feature importance scores in descending order
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Select top 2 features based on importance scores
top_features = [feat for feat, _ in sorted_importance[:1]]
t1=time()
# Print top 2 features
print("Top 1 features:", top_features)
print(t1-t0)
E = top_features



###########RFE with random forest

from time import time
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)

y = data['KilosEntrados']

X = data.drop(['KilosEntrados'], axis=1)


model3 = RandomForestRegressor(n_estimators=100, random_state=42)

t0=time.time()
selector = RFE(estimator=model3, n_features_to_select=1)
selector.fit_transform(X,y)

feature_names = np.array(X.columns)
E=feature_names[selector.get_support()]
t1=time.time()
print("time = ",  t1-t0)
print(E)









##### Lasso regularization

# implement algorithm
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
model3 = LinearSVC(penalty= 'l1', C = 0.05, dual=False)
model3.fit(variables,data.loc[:,'KilosEntrados'])
# select features using the meta transformer
selector = SelectFromModel(estimator = model3, prefit=True)
X_new = selector.transform(variables)
X_new.shape[1]
# names of selected features
feature_names = np.array(variables.columns)
feature_names[selector.get_support()]

new_data=np.zeros((df.shape[0],X_new.shape[1]))
# df_ = pd.DataFrame(index=61, columns=cl_num+1)
for i in range(X_new.shape[1]):
     new_data[:,i]=variables.iloc[:,int(feature_names[selector.get_support()][i])]
     
# new_data[:,X_new.shape[1]]=data2.loc[:,'KilosEntrados']
# new_data=pd.DataFrame(new_data)
# new_data2=new_data.loc[:, new_data.columns != 'KilosEntrados']

model3 = LinearRegression().fit(new_data,data2.loc[:,'KilosEntrados']) 
y_pred = model3.predict(new_data)
mean_absolute_error(data.loc[:,'KilosEntrados'], y_pred)





#######################Granger Causality-based Feature Selection" (GCFS)

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import statsmodels.tsa.stattools as ts
from itertools import combinations
from time import time
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
# Load the multivariate time series data
# data = pd.read_csv('multivariate_time_series.csv')
data.index=pd.to_datetime(data['FechaRecepcion'])
data.drop(columns = 'FechaRecepcion', inplace=True)
# Define the target variable
y = data['KilosEntrados']

# Define the candidate predictor variables
X = data.drop(['KilosEntrados'], axis=1)
t0=time()
# Define the maximum lag for Granger Causality
maxlag = 3

# Perform Granger Causality-based Feature Selection (GCFS)
pvals = np.zeros((X.shape[1], X.shape[1]))

# Loop over all pairs of predictor variables
for i, j in combinations(range(X.shape[1]), 2):
    pvals[i, j] = ts.grangercausalitytests(np.column_stack((X.iloc[:, i], X.iloc[:, j])),
                                           maxlag=maxlag, verbose=False)[maxlag][0]['params_ftest'][1]
    pvals[j, i] = pvals[i, j]

# Create a dataframe of p-values
pvals_df = pd.DataFrame(pvals, columns=X.columns, index=X.columns)



# Select the top two features based on the p-values
n_features = 1  # set the number of features to select
selected_features = pvals_df.max().sort_values().index[:n_features]

t1=time()
# threshold = 0.7  # set the threshold for p-values
# selected_features = pvals_df.columns[pvals_df.max() < threshold]

# Print the selected features
print('Selected Features:', selected_features)
print(t1-t0)

# # Apply Lasso regression to select the most important features
# lasso = LassoCV(cv=10)
# selector = SelectFromModel(lasso)
# selector.fit(X, y)
# selected_features = X.columns[selector.get_support()]

# # Print the selected features
# print('Selected Features:', selected_features)


##########################################





























#