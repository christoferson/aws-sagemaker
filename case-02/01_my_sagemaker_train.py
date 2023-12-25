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

session = boto3.Session(
    aws_access_key_id=config.aws["aws_access_key_id"],
    aws_secret_access_key=config.aws["aws_secret_access_key"],
    region_name=config.aws["region_name"],
)

s3_client = session.client("s3")


sagemaker_session = sagemaker.Session(boto_session=session)