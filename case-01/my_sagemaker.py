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

"""
boto_session = boto3.session.Session(region_name=config.aws["region_name"])
sagemaker_client = boto_session.client(service_name='sagemaker', region_name=config.aws["region_name"])
sagemaker_session = sagemaker.session.Session(boto_session=boto_session, sagemaker_client=sagemaker_client)
"""
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
sagemaker_bucket_prefix = config.aws["sagemaker_bucket_prefix"] #"linear"
sagemaker_role_arn  = config.aws["sagemaker_role_arn"]
#execution_role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
#print(execution_role)

##

salary_df = pd.read_csv("case-01/salary.csv")

X = salary_df[['YearsExperience']]
y = salary_df[['Salary']]

#print(X.shape)
#print(y.shape)

# Convert Data to float needed by sklearn
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

# Split the data to Test and Train Data. Train Data at 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

y_train = y_train[:,0]
y_test = y_test[:,0]

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)
buf.seek(0)

s3_client.upload_fileobj(buf, sagemaker_bucket_name, "{}/train/linear-train-data".format(sagemaker_bucket_prefix))
s3_train_data = "s3://{}/linear/train/linear-train-data".format(sagemaker_bucket_name)
print("Tain Data:{}".format(s3_train_data))


buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_test, y_test)
buf.seek(0)

s3_client.upload_fileobj(buf, sagemaker_bucket_name, "linear/test/linear-test-data")
print("Test Data: s3://{}/linear/test/linear-test-data".format(sagemaker_bucket_name))


output_location = 's3://{}/{}/output'.format(sagemaker_bucket_name, sagemaker_bucket_prefix)
print('Training artifacts will be uploaded to: {}'.format(output_location))

container = get_image_uri(config.aws["region_name"], "linear-learner")

"""
linear = sagemaker.estimator.Estimator(container,
                                       sagemaker_role_arn, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.c4.xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session)
"""
linear = sagemaker.estimator.Estimator(container,
                                       sagemaker_role_arn, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.c4.xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session,
                                       train_use_spot_instances = True,
                                       train_max_run = 300,
                                       train_max_wait = 600)

linear.set_hyperparameters(feature_dim = 1,
                           predictor_type = 'regressor',
                           mini_batch_size = 5,
                           epochs = 5,
                           num_models = 32,
                           loss = 'absolute_loss')

# Now we are ready to pass in the training data from S3 to train the linear learner model

linear.fit({'train': s3_train_data})