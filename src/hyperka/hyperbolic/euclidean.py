from abc import ABC
import torch
from src.hyperka.hyperbolic.manifold import Manifold


class EuclideanManifold(Manifold, ABC):
    __slots__ = ["max_norm"]
    name = "euclidean"

    def __init__(self, max_norm=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def normalize(self, u):
        d = u.shape[-1]
        torch.reshape(u, [-1, d])

        # TODO: torch中如何梯度裁剪还没有实现
        # return tf.clip_by_norm(tf.reshape(u, [-1, d]), self.max_norm, axes=0)

    def distance(self, u, v):
        return torch.sum(torch.pow((u - v), 2), dim=1)

    def exp_map(self, p, d_p, normalize=False, lr=None, out=None):
        if lr is not None:
            d_p = d_p * -lr
        if out is None:
            out = p
        out = out + d_p
        if normalize:
            self.normalize(out)
        return out

    def log_map(self, p, d_p, out=None):
        return p - d_p
