import os

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    sp_tensor = torch.sparse_coo_tensor(indices=[[0, 0, 1, 2], [2, 3, 2, 3]], values=[1, 1, 1, 1],
                                        size=(4, 4)).coalesce()
    dense_tensor = torch.tensor(data=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    masked_tensor = dense_tensor.sparse_mask(sp_tensor)
    print(masked_tensor)
    os.system("pause")

    t1 = torch.tensor([1, -1])
    t2 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(torch.matmul(t1, t2))
    t2 = t2.reshape((8, 1))
    print(t2)
    s = {(1, 2), (3, 4), (5, 6)}
    c = [element[0] for element in s]
    v = [element[1] for element in s]
    print(c)
    print(v)
    s.add((5, 2))
    print(s)
    os.system("pause")
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
