import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import config
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# read the csv file 
insurance_df = pd.read_csv('case-02/insurance.csv')

# check if there are any Null values
isnullsum = insurance_df.isnull().sum()
print(f"{isnullsum}")

# Check the dataframe info
insurance_df.info()

# Detect Non Numeric
#non_numeric_columns = insurance_df[insurance_df.applymap(lambda x: not pd.to_numeric(x, errors='coerce').notna().all())]
#non_numeric_columns = insurance_df.select_dtypes(exclude=np.number)
#print(non_numeric_columns)

# Convert to numberic
#insurance_df = insurance_df.apply(pd.to_numeric, errors='coerce')
#insurance_df.info()

# Grouping by region to see any relationship between region and charges
# Seems like south east region has the highest charges and body mass index
df_region = insurance_df.groupby(by='region').mean("charges")
#df_region = insurance_df.groupby(by='region')['charges'].mean()
#df_region = insurance_df.groupby(by='region').mean().reset_index()
print(f"{df_region}")


# Check unique values in the 'sex' column
unique_sex = insurance_df['sex'].unique()
print(unique_sex)

# convert categorical variable sex to numerical. male=0, female=1
insurance_df['sex'] = insurance_df['sex'].apply(lambda x: 0 if x == 'female' else 1)
print(insurance_df.head())

# Check the unique values in the 'smoker' column
unique_smoker = insurance_df['smoker'].unique()
print(unique_smoker)

# Convert categorical variable smoker to numerical. no=0 yes=1
insurance_df['smoker'] = insurance_df['smoker'].apply(lambda x: 0 if x == 'no' else 1)
print(insurance_df.head())

# Check unique values in 'region' column
unique_region = insurance_df['region'].unique()
print(unique_region)

# Convert the region to column flags
region_dummies = pd.get_dummies(insurance_df['region'], drop_first = True)
print(region_dummies)

insurance_df = pd.concat([insurance_df, region_dummies], axis = 1)
print(insurance_df.head())

# Let's drop the original 'region' column 
insurance_df.drop(['region'], axis = 1, inplace = True)
print(insurance_df.head())

histogram = insurance_df[['age', 'sex', 'bmi', 'children', 'smoker', 'charges']].hist(bins = 30, figsize = (20,20), color = 'r')
print(histogram)

#sns.pairplot(insurance_df)

# Plot a linear line for age vs charges
sns.regplot(x = 'age', y='charges', data=insurance_df)
plt.show()

# Plot a linear line for bmi vs charges
#sns.regplot(x = 'bmi', y='charges', data=insurance_df)
#plt.show()

correlation = insurance_df.corr()
print(correlation)

plt.figure(figsize=(10, 10))
heatmap = sns.heatmap(correlation, annot=True)
print(heatmap)
plt.show()


#### Training

print(insurance_df.columns)

X = insurance_df.drop(columns =['charges'])
y = insurance_df['charges']
print(X)
print(y)

print(X.shape)
print(y.shape)

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

y = y.reshape(-1,1)

print(X)
print(X.shape)
print(y)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
print(X_train.shape)
print(X_test.shape)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

####

regresssion_model_sklearn = LinearRegression()
regresssion_model_sklearn.fit(X_train, y_train)

regresssion_model_sklearn_accuracy = regresssion_model_sklearn.score(X_test, y_test)
print(regresssion_model_sklearn_accuracy)

y_predict = regresssion_model_sklearn.predict(X_test)

y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)

k = X_test.shape[1]
n = len(X_test)
print(n)

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)

#print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
print('RMSE =',RMSE, '\nMSE =',MSE) 