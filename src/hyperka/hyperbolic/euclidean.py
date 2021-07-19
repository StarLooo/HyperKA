import tensorflow as tf
from abc import ABC
import torch
from hyperka.hyperbolic.manifold import Manifold
import util


# modified by lxy
class EuclideanManifold(Manifold, ABC):
    __slots__ = ["max_norm"]
    name = "euclidean"

    def __init__(self, max_norm=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    # modified by lxy
    # TODO: I don't know how to transform this tensorflow code to torch code
    def normalize(self, u):
        # I think here shouldn't have ".value"
        # d = u.shape[-1].value
        d = u.shape[-1]
        torch.reshape(u, [-1, d])

        # TODO: I don't know how to change tf.clip_by_norm into torch version
        # return tf.clip_by_norm(tf.reshape(u, [-1, d]), self.max_norm, axes=0)

    # modified by lxy
    def distance(self, u, v):
        changed = torch.sum(torch.pow((u - v), 2), dim=1)
        if util.DEBUG:
            origin = tf.reduce_sum(tf.pow((u - v), 2), axis=1)
            util.judge_change_equal(origin, changed)
        return changed

    # lxy: these 2 func may be useless
    # def pnorm(self, u, dim=None):
    #     return tf.sqrt(tf.reduce_sum(u * u, axis=dim))
    #
    # def rgrad(self, p, d_p):
    #     return d_p

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


if __name__ == '__main__':
    a = tf.constant([[1., 2., 3., 4., 5.]])
    b = tf.constant([[0., 0., 0., 0., 0.]])
    c = tf.pow(a - b, 2)
    print(c)
    euclidean = EuclideanManifold(max_norm=1)
    print(euclidean.distance(a, b))
