import numpy as np
import matplotlib.pyplot as plt

def dice_rolls(n):
    dice1 = np.random.randint(1, 7, size=n)
    dice2 = np.random.randint(1, 7, size=n)
    return dice1 + dice2

def plot_histogram(results, n):
    h, bins = np.histogram(results, range=(2, 13))
    plt.bar(bins[:-1], h/n)
    plt.title(f"Histogram of sums of two dice (n = {n})")
    plt.xlabel("Sum of two dice")
    plt.ylabel("Relative frequency")
    plt.show()

values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in values:
    results = dice_rolls(n)
    plot_histogram(results, n)
