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

# read the csv file 
insurance_df = pd.read_csv('case-02/insurance.csv')

# check if there are any Null values
insurance_df.isnull().sum()

# Check the dataframe info
insurance_df.info()