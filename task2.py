import pandas as pd
import matplotlib.pyplot as plt

data = {
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "y": [-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78]
}

df = pd.DataFrame(data)
plt.figure(figsize=(7,5))
plt.scatter(df["x"], df["y"], marker="+", color="b", s=50)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot")
plt.grid(True)
plt.show()
