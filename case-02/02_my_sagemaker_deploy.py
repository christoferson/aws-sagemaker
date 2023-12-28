import boto3
import sagemaker
from sagemaker import Session
import sys
import os
import io
import numpy as np
import seaborn as sns
import pandas as pd
import sagemaker.amazon.common as smac
from sagemaker.amazon.amazon_estimator import get_image_uri
from sklearn.model_selection import train_test_split
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current) 
sys.path.append(parent)

import config


session = boto3.Session(
    aws_access_key_id=config.aws["aws_access_key_id"],
    aws_secret_access_key=config.aws["aws_secret_access_key"],
    region_name=config.aws["region_name"],
)

s3_client = session.client("s3")


sagemaker_session = sagemaker.Session(boto_session=session)
print(sagemaker_session._region_name)
print(sagemaker_session.account_id())

sagemaker_bucket_name = config.aws["sagemaker_bucket_name"]
sagemaker_bucket_prefix = config.case02["sagemaker_bucket_prefix"] #"linear-case-02"
sagemaker_role_arn  = config.aws["sagemaker_role_arn"]


###

insurance_df = pd.read_csv('case-02/insurance.csv')

insurance_df['sex'] = insurance_df['sex'].apply(lambda x: 0 if x == 'female' else 1)
print(insurance_df.head())

# Convert categorical variable smoker to numerical. no=0 yes=1
insurance_df['smoker'] = insurance_df['smoker'].apply(lambda x: 0 if x == 'no' else 1)
print(insurance_df.head())

# Convert the region to column flags
region_dummies = pd.get_dummies(insurance_df['region'], drop_first = True)
print(region_dummies)

insurance_df = pd.concat([insurance_df, region_dummies], axis = 1)
print(insurance_df.head())

# Let's drop the original 'region' column 
insurance_df.drop(['region'], axis = 1, inplace = True)
print(insurance_df.head())

insurance_df.info()
print(insurance_df)

print(insurance_df.columns)

X = insurance_df.drop(columns =['charges'])
y = insurance_df['charges']
#print(X)
#print(y)

print(X.shape)
print(y.shape)

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

y = y.reshape(-1,1)

#print(X)
print(X.shape)
#print(y)
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

###

container = get_image_uri(config.aws["region_name"], "linear-learner")

linear = sagemaker.estimator.Estimator.attach(
        training_job_name = "linear-learner-2023-12-28-04-45-28-750",
        sagemaker_session = sagemaker_session)

linear_regressor = linear.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

# linear_regressor.content_type = 'text/csv'
linear_regressor.serializer = CSVSerializer()
linear_regressor.deserializer = JSONDeserializer()

# making prediction on the test data
print(f"X_test:{X_test}")
result = linear_regressor.predict(X_test)
print(result)

predictions = np.array([r['score'] for r in result['predictions']])
print(predictions)

print(predictions.shape)


y_predict_orig = scaler_y.inverse_transform(predictions)
y_test_orig = scaler_y.inverse_transform(y_test)


RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# Delete the end-point
#linear_regressor.delete_endpoint()