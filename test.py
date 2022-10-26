import torch


x = torch.Tensor([[1, 2], [3, 4]])
x[1, []] = 0

print(x)