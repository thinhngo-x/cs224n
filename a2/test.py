import numpy as np

a = np.array([[1, 0, 0],
              [2, 0, 0],
              [3, 0, 0]])
idx = [0, 0]
np.add.at(a, idx, 1)
print(a)
