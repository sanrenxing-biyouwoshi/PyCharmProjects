import torch

a = torch.tensor([[1., 2, 3, 4, 5, ], [6, 7, 8, 9, 10]])
b = torch.tensor([0, 1])
c = a.index_select(1, b)
d = a.index_select(0, b)
print(c)
print(d)
