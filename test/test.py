import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    t = torch.zeros(size=(3, 5))
    print(t.T)
    print(t[2].shape)
    print(t[2, :].shape)
    print(t[2].unsqueeze(dim=0))
    set_1 = {1, 2, 3}
    set_2 = {4, 5, 6}
    print(set_1 | set_2)
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    b = torch.tensor([3., 4., 5.], requires_grad=False)
    c = a + b
    print(c.requires_grad)
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
