import boto3
import sagemaker
import sys
import os
import json

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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




sagemaker_runtime_client = session.client('sagemaker-runtime')

custom_attributes = "c002b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
endpoint_name = "linear-learner-2023-12-28-14-56-51-106"                                       # Your endpoint name.
content_type = "text/csv"                                        # The MIME type of the input data in the request body.
accept = "application/json"                                              # The desired MIME type of the inference in the response.
payload = b"19,0,35.15,0,0,1,0,0\n56,0,9.82,0,0,0,0,1"  #19,female,35.15,0,no,northwest,2134.9015  56,female,39.82,0,no,southeast,11090.7178


response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, 
    CustomAttributes=custom_attributes, 
    ContentType=content_type,
    Accept=accept,
    Body=payload
)

print(response)

body = response['Body']
result = json.load(body)

predictions = np.array([r['score'] for r in result['predictions']])

print(predictions)
print(predictions.shape)

##

