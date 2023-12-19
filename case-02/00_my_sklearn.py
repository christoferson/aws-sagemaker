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