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
#from sagemaker.predictor import csv_serializer, json_deserializer
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import matplotlib.pyplot as plt

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
sagemaker_bucket_prefix = "linear"
sagemaker_role_arn  = config.aws["sagemaker_role_arn"]


###
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

y_train = y_train[:,0]
y_test = y_test[:,0]


###

container = get_image_uri(config.aws["region_name"], "linear-learner")

"""
linear = sagemaker.estimator.Estimator(container,
                                       sagemaker_role_arn, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.c4.xlarge',
                                       output_path = "s3://{}/{}/output".format(sagemaker_bucket_name, sagemaker_bucket_prefix),
                                       sagemaker_session = sagemaker_session,
                                       train_use_spot_instances = True,
                                       train_max_run = 300,
                                       train_max_wait = 600)
"""
linear = sagemaker.estimator.Estimator.attach(
        training_job_name="linear-learner-2023-12-17-16-27-36-884",
        sagemaker_session=sagemaker_session)

"""
linear.set_hyperparameters(feature_dim = 1,
                           predictor_type = 'regressor',
                           mini_batch_size = 5,
                           epochs = 5,
                           num_models = 32,
                           loss = 'absolute_loss')
"""

linear_regressor = linear.deploy(initial_instance_count = 1,
                                          instance_type = 'ml.m4.xlarge')

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

# VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'gray')
plt.plot(X_test, predictions, color = 'red')
plt.xlabel('Years of Experience (Testing Dataset)')
plt.ylabel('salary')
plt.title('Salary vs. Years of Experience')



# Delete the end-point
#linear_regressor.delete_endpoint()