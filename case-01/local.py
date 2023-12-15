import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

salary_df = pd.read_csv("case-01/salary.csv")

#print(salary_df)

print(salary_df.head(3))

print(salary_df.tail(3))

print(salary_df['Salary'].max())

# check if there are any Null values
hm = sns.heatmap(salary_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

# Check the dataframe info
salary_df.info()

# Statistical summary of the dataframe
stats_df = salary_df.describe()
print(stats_df)

max = salary_df[salary_df['Salary'] == salary_df['Salary'].max()]
print(f"Max: {max}")

min = salary_df[salary_df['Salary'] == salary_df['Salary'].min()]
print(f"Min: {min}")

hist = salary_df.hist(bins = 30, figsize = (20,10), color = 'r')
print(hist)