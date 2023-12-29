import boto3
import sagemaker
import sys
import os
import json

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


sagemaker_runtime_client = session.client('sagemaker-runtime')

custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
endpoint_name = "linear-learner-2023-12-18-13-27-16-419"                                       # Your endpoint name.
content_type = "text/csv"                                        # The MIME type of the input data in the request body.
accept = "application/json"                                              # The desired MIME type of the inference in the response.
payload = b"5.1" 
payload = b"8.7" #102829.125

response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, 
    CustomAttributes=custom_attributes, 
    ContentType=content_type,
    Accept=accept,
    Body=payload
)

print(response)

body = response['Body']
predictions = json.load(body)

print(predictions)