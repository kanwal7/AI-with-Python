import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

df = pd.read_csv("Auto.csv")
print("First few rows of dataset:")
print(df.head())

y = df['mpg']
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']]

print("\nMissing values in dataset:")
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
alphas = np.logspace(-3, 3, 30) 
ridge_scores = []
lasso_scores = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test_scaled)))
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test_scaled)))

plt.figure(figsize=(8,5))
plt.semilogx(alphas, ridge_scores, marker='o', label='Ridge R2')
plt.semilogx(alphas, lasso_scores, marker='s', label='Lasso R2')
plt.xlabel('Alpha (log scale)')
plt.ylabel('R2 Score (Test Data)')
plt.title('Ridge vs Lasso Regression Performance on Auto.csv')
plt.legend()
plt.grid(True)
plt.show()

best_alpha_ridge = alphas[np.argmax(ridge_scores)]
best_r2_ridge = max(ridge_scores)
best_alpha_lasso = alphas[np.argmax(lasso_scores)]
best_r2_lasso = max(lasso_scores)
print(f"\nBest Ridge = {best_alpha_ridge:.4f} with R² = {best_r2_ridge:.4f}")
print(f"Best Lasso = {best_alpha_lasso:.4f} with R² = {best_r2_lasso:.4f}")

lasso_best = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_best.fit(X_train_scaled, y_train)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_best.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nLasso Coefficients:")
print(coef_df)

"""
Findings:
- R² score shows how well the model predicts 'mpg' on unseen data.
- Ridge often handles multicollinearity better; Lasso performs variable selection.
- Non-zero coefficients in Lasso highlight the most influential predictors.
- The alpha with highest R² is the optimal regularization strength for each model.
"""