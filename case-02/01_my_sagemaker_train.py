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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

import config

session = boto3.Session(
    aws_access_key_id=config.aws["aws_access_key_id"],
    aws_secret_access_key=config.aws["aws_secret_access_key"],
    region_name=config.aws["region_name"],
)

s3_client = session.client("s3")


sagemaker_session = sagemaker.Session(boto_session=session)

sagemaker_bucket_name = config.aws["sagemaker_bucket_name"]
sagemaker_bucket_prefix = config.case02["sagemaker_bucket_prefix"] #"linear-case-02"
sagemaker_role_arn  = config.aws["sagemaker_role_arn"]

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

buf = io.BytesIO() # create an in-memory byte array (buf is a buffer I will be writing to)
smac.write_numpy_to_dense_tensor(buf, X_train, y_train.reshape(-1))
buf.seek(0) 

s3_client.upload_fileobj(buf, sagemaker_bucket_name, "{}/train/linear-train-data".format(sagemaker_bucket_prefix))
s3_train_data = "s3://{}/{}/train/linear-train-data".format(sagemaker_bucket_name, sagemaker_bucket_prefix)
print("Tain Data:{}".format(s3_train_data))

# Set test data
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_test, y_test.reshape(-1))
buf.seek(0)

s3_client.upload_fileobj(buf, sagemaker_bucket_name, "{}/test/linear-test-data".format(sagemaker_bucket_prefix))
print("Test Data: s3://{}/{}/test/linear-test-data".format(sagemaker_bucket_name, sagemaker_bucket_prefix))

output_location = 's3://{}/{}/output'.format(sagemaker_bucket_name, sagemaker_bucket_prefix)
print('Training artifacts will be uploaded to: {}'.format(output_location))

container = get_image_uri(config.aws["region_name"], "linear-learner")

# We have pass in the container, the type of instance that we would like to use for training 
# output path and sagemaker session into the Estimator. 
# We can also specify how many instances we would like to use for training

linear = sagemaker.estimator.Estimator(container,
                                       sagemaker_role_arn, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.c4.xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session,
                                       train_use_spot_instances = True,
                                       train_max_run = 300,
                                       train_max_wait = 600)


# We can tune parameters like the number of features that we are passing in, type of predictor like 'regressor' or 'classifier', mini batch size, epochs
# Train 32 different versions of the model and will get the best out of them (built-in parameters optimization!)

linear.set_hyperparameters(feature_dim = 8,
                           predictor_type = 'regressor',
                           mini_batch_size = 100,
                           epochs = 100,
                           num_models = 32,
                           loss = 'absolute_loss')


# Now we are ready to pass in the training data from S3 to train the linear learner model

linear.fit({'train': s3_train_data})
