import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("bank.csv", delimiter=';')

print("Variables in the dataset:")
print(df.info())
print("Data type of each variable:")
print(df.dtypes)

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
df3 = pd.get_dummies(df2, columns=['y', 'job', 'marital', 'default', 'housing', 'poutcome'])
corr_matrix = df3.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Correlation Coefficients for df3 Variables")
plt.show()

y = df3['y_yes']
X = df3.drop(['y_yes', 'y_no'], axis=1) 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)


y_pred_log = log_reg.predict(X_test_scaled)

cm_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)

print("Logistic Regression Accuracy:", round(acc_log, 4))
print("Confusion Matrix:\n", cm_log)

plt.figure(figsize=(5,4))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("KNN (k=3) Accuracy:", round(acc_knn, 4))
print("Confusion Matrix:\n", cm_knn)

plt.figure(figsize=(5,4))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("KNN (k=3) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("=== Model Performance Comparison ===")
print(f"Logistic Regression Accuracy: {acc_log:.4f}")
print(f"KNN (k=3) Accuracy: {acc_knn:.4f}\n")

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log, target_names=['no', 'yes']))

print("KNN (k=3) Report:")
print(classification_report(y_test, y_pred_knn, target_names=['no', 'yes']))

"""
Findings:
- Logistic Regression achieved slightly higher accuracy and better precision for the 'yes' class.
- KNN (k=3) performed slightly worse and is more sensitive to scaling and noise.
- Logistic Regression is generally preferred for interpretability and stable performance on this dataset.
"""
