import torch
import numpy as np
# import tensorflow as tf

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
DEBUG = False  # define by lxy, switch to debug model


# add by lxy
def judge_change_equal(origin_tf_tensor, changed_torch_tensor):
    origin = origin_tf_tensor.numpy()
    changed = changed_torch_tensor.detach().numpy()
    if not np.allclose(origin, changed, atol=1e-3, rtol=1e-3, equal_nan=True):
        print("origin:", origin)
        print("changed:", changed)
        raise RuntimeError("detect error on change from tensorflow to pytorch!")


# modified by lxy
# Real x, not vector!
def util_atanh(x):
    changed = torch.atanh(torch.minimum(x, torch.full_like(x, 1. - EPS)))
    # if DEBUG:
    #     origin = tf.atanh(tf.minimum(x, 1. - EPS))  # Only works for positive real x.
    #     judge_change_equal(origin, changed)
    return changed


# modified by lxy
# Real x, not vector!
def util_tanh(x):
    changed = torch.tanh(torch.minimum(torch.maximum(x, torch.full_like(x, -MAX_TANH_ARG)),
                                       torch.full_like(x, MAX_TANH_ARG)))
    # if DEBUG:
    #     origin = tf.tanh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))
    #     judge_change_equal(origin, changed)
    return changed


# modified by lxy
def util_dot(x, y):
    changed = torch.sum(input=x * y, dim=1, keepdim=True)
    # if DEBUG:
    #     origin = tf.reduce_sum(x * y, axis=1, keepdims=True)
    #     judge_change_equal(origin, changed)
    return changed


# modified by lxy
def util_norm(x):
    changed = torch.norm(input=x, p=2, dim=-1, keepdim=True)
    # if DEBUG:
    #     origin = tf.norm(x, ord=2, axis=-1, keepdims=True)
    #     judge_change_equal(origin, changed)
    return changed


# test for this util
if __name__ == '__main__':
    test_loop_num = 1000
    # test for tf_tanh()
    for _ in range(test_loop_num + 1):
        x = torch.randn(size=[1, 1])
        print("x:", x, "util_tanh(x):", util_tanh(x))

    # test for tf_atanh()
    for _ in range(test_loop_num + 1):
        x = torch.randn(size=[1, 1])
        print("x:", x, "util_atanh(x):", util_tanh(x))

    # test for tf_dot()
    for _ in range(test_loop_num + 1):
        x, y = torch.randn(size=[2, 3]), torch.randn(size=[2, 3])
        print("x:", x)
        print("y:", y)
        print("util_dot(x, y):", util_dot(x, y))

    # test for tf_norm()
    for _ in range(test_loop_num + 1):
        x = torch.randn(size=[2, 3])
        print("x:", x)
        print("util_norm(x):", util_norm(x))
