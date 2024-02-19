### Case 03

Forecast weekly retail stores sales based on historical data.
Weekly sales from different departments belonging to 45 differerent stores
Data contains holidays and promotional markdowns offered by various stores and several departments throughout the year
Markdowns are crucial to promoting sales especially before key events such as Super Bowl, Christmas and Thanksgiving.


##### Bias & Variance Tradeoff

Underfitting
Overfitting - Algorithm models the training data too perfectly. e.g. memorizing the answers. Fails to generalize or performs poorly on actual data.

Linear Model - High Bias, Low Variance, Good Generalization
High Order Polynomial  - Low Bias, High Variance, Overfitted


##### Ridge Regression

Overfitting - Algorithm is very good in predicting training data but performs poorly on test data.

Ridge Regression - Attempt to increase the bias to improve variance. Add Alpha times Square of Slope to decouple independent variable.


###### Boosting

Works by training and testing multiple models in succession where the current model learns or improves from the previous model's mistakes or residual. The training and testing cycle continues until the maximum iteration is met or satisfactory result is achieved.


###### Ensemble Learning

Combine multiple weak or average algorithms to produce a more robust output. Similar to Blind men and the elephant.

Decision Tree Ensemble - Data is split according to a certain condition / feature.

Ex. To determine if a certain individual will play a game or not, we create the ff. decision trees:
Age < 20 - Yes (2 pts), No (-1 pts)
Use Computer Daily - Yes (0.8 pts), No (-0.8 pts)
Gender Male - Yes (0.5 pts) No (0.2 pts)
To see how probable a 7 year old boy who uses the computer on weekends only, we can calculate as follows:
2 + -0.8 + 0.5 = 1.7 pts


##### XGBoost

Extreme Gradient Boosting - supervised learning algorithm that implements gradient bossted trees algorithm

Algorithm works by combining an ensemble of predictions from several weak models.

Pros:

- No need to do feature scaling

- Can work well with missing data

- Robust to outliers in the data

- Can work well for both regression and classification

- Computationally efficient and fast

- Can do training process on many machines

Cons:

- Poor extrapolation capabilities - Out of bound inference can produce erratic results

- Need extensive tuning

- Slow training

##### Gradient Boosted Trees

Algorithm
1. Create an ensemble tree from the input features
2. Classify the data set per the ensemble tree and add the residual as leaf nodes.
   Initially, since there is no previous prediction yet, use a arbitrary constant as the prediction and calculate the residual by subtracting the constant from the Truth Value.
3. If there are multiple leaf nodes, calculate the average to reduce to a single value.
4. For each record, calculate the new prediction value
To calculate the prediction values, Previous prediction + Learning Rate e.g. 0.2 x Corresponding Leaf Node Value
Learning Rate is use to slow down the pace, add gradient to the boosting. This is to avoid overfitting.
5. Repeat until satisfied

##### Sagemaker XGBoost Implementation

- Uses tabular data for input and output
  - A row represents an observation
  - A column represents output or target label
  - The rest of the columns represent input features

- Supported file format. Note protobuf not supported
  - CSV
  - libsvm

- XGBoost is memory intensive and not so much on compute. M4 general purpose is recommended.


