from matplotlib import pyplot as plt
import pandas as pd
import numpy as np  
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('50_Startups.csv',delimiter=',')
print("Variables in the dataset:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

corr = df.corr(numeric_only=True)
print("\nCorrelation matrix:")
print(corr)

df_var = pd.get_dummies(df, columns=['State'], drop_first=True)
features = ['R&D Spend', 'Administration', 'Marketing Spend']

for val in features:
    plt.figure()
    sns.scatterplot(x=df[val], y=df['Profit'])
    plt.title(f'{val} vs Profit')
    plt.show()

X = df_var.drop('Profit', axis=1)
y = df_var['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"\nTraining RMSE: {rmse_train:.2f}")
print(f"Testing RMSE: {rmse_test:.2f}")
print(f"Training R2: {r2_train:.2f}")
print(f"Testing R2: {r2_test:.2f}")

"""
Findings:
  - R² indicates how much of the variation in Profit is explained by the model.
  - RMSE gives an idea of average prediction error in Profit units.
  - Usually, R&D Spend dominates as the key predictor.
"""