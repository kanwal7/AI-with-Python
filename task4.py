import pandas as pd
import numpy as np

A = pd.DataFrame([[1, 2, 3],
                       [0, 1, 4],
                       [5, 6, 0]], dtype=float)

A_inv = pd.DataFrame(np.linalg.inv(A.values), columns=A.columns, index=A.index)

print("Inverse of A:")
print(A_inv)
print("\nA @ A_inv:")
print(pd.DataFrame(A.values @ A_inv.values, columns=A.columns, index=A.index))
print("\nA_inv @ A:")
print(pd.DataFrame(A_inv.values @ A.values, columns=A.columns, index=A.index))
