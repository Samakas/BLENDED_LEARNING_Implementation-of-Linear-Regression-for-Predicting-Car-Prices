import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

df = pd.read_csv("CarPrice_Assignment (1).csv")

print(df.head())

df = df.dropna()

X = df[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]

y = df['price']


# Standardize the features
SS = StandardScaler()

X = SS.fit_transform(X)
y = SS.fit_transform(y.values.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Check model assumptions
# 1. Linearity Assumption
plt.figure(figsize=(10, 6))
for i, col in enumerate(['horsepower', 'curbweight', 'enginesize', 'highwaympg']):
    plt.subplot(2, 2, i+1)
    plt.scatter(df[col], df['price'])
    plt.xlabel(col)
    plt.ylabel('Price')
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.show()

# 2. Homoscedasticity
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Prices')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# 3. Normality
plt.figure(figsize=(10, 6))
plt.hist(y_test - y_pred, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# 4. Multicollinearity
corr_matrix = df[['horsepower', 'curbweight', 'enginesize', 'highwaympg']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


