import torch
import numpy as np

print("hello")

t2 = torch.tensor([1., 2, 3, 4])
print(t2)

x = np.array([[1, 2], [3, 4]])
print(x)

y = torch.from_numpy(x)
print(y)
