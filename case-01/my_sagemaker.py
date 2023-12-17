import boto3
import sagemaker
from sagemaker import Session
import sys
import os

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

sagemaker_session = sagemaker.Session(boto_session=session)
print(sagemaker_session._region_name)
print(sagemaker_session.account_id())

#execution_role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
#print(execution_role)