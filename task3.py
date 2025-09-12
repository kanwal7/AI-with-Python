import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('weight-height(1).csv')
length_in = df.iloc[:, 1]
weight_lb = df.iloc[:, 2]
length_cm = length_in * 2.54
weight_kg = weight_lb * 0.453592
mean_length = length_cm.mean()
mean_weight = weight_kg.mean()

print(f"Mean height: {mean_length:.2f} cm")
print(f"Mean weight: {mean_weight:.2f} kg")

plt.hist(length_cm, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Student Heights')
plt.xlabel('Height (cm)')
plt.ylabel('Number of Students')
plt.show()
