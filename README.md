# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries:
Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset:
Load the dataset containing car prices and relevant features.
3. Data Preprocessing:
Handle missing values and perform feature selection if necessary.
4. Split Data:
Split the dataset into training and testing sets.
5. Train Model:
Create a linear regression model and fit it to the training data.
6. Make Predictions:
Use the model to make predictions on the test set.
7. Evaluate Model:
Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions:
Plot residuals to check for homoscedasticity, normality, and linearity.
9. Output Results:
Display the predictions and evaluation metrics.
Program to implement linear regression model for predicting car prices and test assumptions.

## Program:
Program to implement linear regression model for predicting car prices and test assumptions.<br>
Developed by: SAMAKASH R S<br>
RegisterNumber:  212223230182
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import numpy as np
import statsmodels.api as sm
df = pd.read_csv('CarPrice_Assignment (1).csv')
df
X = df[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

SS = StandardScaler()

X_train_scaled = SS.fit_transform(X_train)
X_test_scaled = SS.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

feature_names = ['horsepower', 'curbweight', 'enginesize', 'highwaympg']

print("="*50)
print("MODEL COEFFICIENTS:")
for feature, coef in zip(feature_names, model.coef_):  # no [0] here
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'intercept':>12}: {model.intercept_:>10.2f}")


print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R-Squared':>12}: {r2_score(y_test, y_pred):>10.2f}")
print("="*50)


# 1. Linearity Check
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity check
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, line_kws={"color": "red"})  # removed lowess=True
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

# 4. Normality of residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")

sm.qqplot(residuals, line="45", fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")

plt.tight_layout()
plt.show()

```

## Output:
![alt text](<Screenshot 2025-04-16 133353-1.png>) 

![alt text](<Screenshot 2025-04-16 133615-1.png>) 

![alt text](<Screenshot 2025-04-16 133621-1.png>) 

![alt text](<Screenshot 2025-04-16 133626-1.png>) 

![alt text](<Screenshot 2025-04-16 133638-1.png>) 

![alt text](<Screenshot 2025-04-16 133700-1.png>) 

![alt text](<Screenshot 2025-04-16 133715-1.png>)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
