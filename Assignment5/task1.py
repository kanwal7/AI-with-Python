import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

df = load_diabetes()
X = pd.DataFrame(df.data, columns=df.feature_names) 
y = df.target

X_feature = X[['bmi','s5']]
X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=42)   
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse_basic = mean_squared_error(y_test, y_pred)
r2_new = r2_score(y_test, y_pred)
print("Model with 'bmi' and 's5':")
print(f"MSE: {mse_basic:.2f}, R2: {r2_new:.2f}")

X_new = X[['bmi','s5','s6']]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.2, random_state=42)
model_new = LinearRegression()  
model_new.fit(X_train_new, y_train_new)
y_pred_new = model_new.predict(X_test_new)
mse_new = mean_squared_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)
print("\nModel with 'bmi','s5' and 's6':")
print(f"MSE: {mse_new:.2f}, R2: {r2_new:.2f}")
"""I added s6 because it represents blood glucose level, 
which is directly related to diabetes severity and 
helps improve prediction accuracy."""

X_all = X
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
model_all = LinearRegression()
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)
mse_all = mean_squared_error(y_test, y_pred_all)
r2_all = r2_score(y_test, y_pred_all)
print("\nModel with all variables:")
print(f"MSE: {mse_all:.2f}, R2: {r2_all:.2f}")
'''Adding more variables can help slightly improve the model’s accuracy, 
but after a point the gains become small because some features are correlated 
or add little new information.'''