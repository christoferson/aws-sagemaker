import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

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

X = salary_df[['YearsExperience']]
y = salary_df[['Salary']]

print(X.shape)
print(y.shape)

# Convert Data to float needed by sklearn
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

# Split the data to Test and Train Data. Train Data at 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

# Train Model
regression_model_sklearn = LinearRegression(fit_intercept=True)
regression_model_sklearn.fit(X_train, y_train)

# Score the model using the Test Data
regression_model_sklearn_accuracy = regression_model_sklearn.score(X_test, y_test)
print(f"Accuracy: {regression_model_sklearn_accuracy}")

print(f"Coefficient: {regression_model_sklearn.coef_}")
print(f"Intercept: {regression_model_sklearn.intercept_}")