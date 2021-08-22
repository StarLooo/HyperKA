import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    print(torch.__version__)
    # a = torch.empty(size=(5, 5), dtype=torch.float64)
    # torch.nn.init.xavier_uniform_(a, gain=1)
    # print(a)
    # a[-1] = torch.empty(size=(5,), dtype=torch.float64)
    # print(a)
    x = torch.tensor([1, 2, 3])
    A = torch.tensor([[1, 2, 3], [3, 4, 5]])
    print(x.shape, A.shape)
    print(torch.mv(A, x))
