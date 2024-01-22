Simple Linear Regression

# Formula

y = mx + b

y - dependent variable
x - independent variable
m - slope, coefficient
b - y intercept, bias

# Evaluate How Good the Formula is

## Regression - Error

Difference of Training Data vs Model Line
Error = Estimated/Predicted Value - Actual/Training Value

## MAE Mean Absolute Error
Average of the magnitude of the errors. Abs(Sum of Errors) / N
If MAE = 0, then the model is perfect
Error increases in Linear Fashion

## MSE Mean Square Error
Average of the square of the errors. Square(Sum of Errors) / N
MSE is generally larger than MAE
Error increases in Quadratic Fashion
Error is heavilly penalized

## RMSE Root Mean Square Error
SquareRoot of MSE
Sandard deviation of the residuals
RMSE is easier to interpret since it matches the units of the output.

## MAPE Mean Absolute Percentage Error
MAE can range from 0 to infinity so an be difficult to interpret
MAPE provides MAE in percentage

## MPE Mean Percentage Error
MAPE without Absolute operation
Gives insight on how many is positive and how many is negative


# Coefficient of Determination

## R2 / R Square Coefficient of Determination
Proportion of variance that has been explained by the independent variables in the model
e.g. If R2=80, that means 80% of the insurance cost is due to the increase in the increase in Age.
Maximum value is 1 - The dependent variable is 100% determined by the independent variable.

## Adjusted R Square
R2 increases as independent variables are added. Can be misleading, especially if the independent variable is unrelated
Adjusted R2 compensates the limitation by adding a penalty when you add an unrelated independent variable
When useless predictors are added, Adusted R2 will decrease. On the otherhand if usefull predictors are added, value will increase.



