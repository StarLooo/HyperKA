import tensorflow as tf
import numpy as np
from hyperka.hyperbolic.euclidean import EuclideanManifold
from hyperka.hyperbolic.util import util_norm, util_tanh, util_atanh
import util
import torch


# modified by lxy
class PoincareManifold(EuclideanManifold):
    name = "poincare"

    def __init__(self, eps=1e-15, projection_eps=1e-5, radius=1.0, **kwargs):
        super(PoincareManifold, self).__init__(**kwargs)
        self.eps = eps
        self.projection_eps = projection_eps
        self.radius = radius  # the radius of the poincare ball
        self.max_norm = 1 - eps
        self.min_norm = eps

    # modified by lxy
    # func to compute Hyperbolic distance
    def distance(self, u, v):
        # TODO: what does the prefix "sq" means?
        changed_sq_u_norm = torch.sum(u * u, dim=-1, keepdim=True)
        changed_sq_v_norm = torch.sum(v * v, dim=-1, keepdim=True)
        changed_sq_u_norm = torch.clip(changed_sq_u_norm, min=0.0, max=self.max_norm)
        changed_sq_v_norm = torch.clip(changed_sq_v_norm, min=0.0, max=self.max_norm)
        changed_sq_dist = torch.sum(torch.pow(u - v, 2), dim=-1, keepdim=True)
        changed_distance = tf.acosh(
            1 + self.eps + (changed_sq_dist / ((1 - changed_sq_u_norm) * (1 - changed_sq_v_norm)) * 2))
        if util.DEBUG:
            origin_sq_u_norm = tf.reduce_sum(u * u, axis=-1, keepdims=True)
            origin_sq_v_norm = tf.reduce_sum(v * v, axis=-1, keepdims=True)
            origin_sq_u_norm = tf.clip_by_value(origin_sq_u_norm, clip_value_min=0.0, clip_value_max=self.max_norm)
            origin_sq_v_norm = tf.clip_by_value(origin_sq_v_norm, clip_value_min=0.0, clip_value_max=self.max_norm)
            origin_sq_dist = tf.reduce_sum(tf.pow(u - v, 2), axis=-1, keepdims=True)
            origin_distance = tf.acosh(
                1 + self.eps + (origin_sq_dist / ((1 - origin_sq_u_norm) * (1 - origin_sq_v_norm)) * 2))
            util.judge_change_equal(origin_distance, changed_distance)
        return changed_distance

    # modified by lxy
    # func to compute mobius addition
    def mobius_addition(self, vectors_u, vectors_v):
        changed_norms_u = self.radius * torch.sum(torch.square(vectors_u), dim=-1, keepdim=True)
        changed_norms_v = self.radius * torch.sum(torch.square(vectors_v), dim=-1, keepdim=True)
        changed_inner_product = self.radius * torch.sum(vectors_u * vectors_v, dim=-1, keepdim=True)
        changed_denominator = 1 + 2 * changed_inner_product + changed_norms_u * changed_norms_v

        changed_denominator = torch.maximum(changed_denominator, torch.full_like(changed_denominator, self.min_norm))

        changed_numerator = (1 + 2 * changed_inner_product + changed_norms_v) * vectors_u + (
                1 - changed_norms_u) * vectors_v
        changed_results = tf.math.divide(changed_numerator, changed_denominator)

        if util.DEBUG:
            origin_norms_u = self.radius * tf.reduce_sum(tf.square(vectors_u), -1, keepdims=True)
            origin_norms_v = self.radius * tf.reduce_sum(tf.square(vectors_v), -1, keepdims=True)
            origin_inner_product = self.radius * tf.reduce_sum(vectors_u * vectors_v, -1, keepdims=True)
            origin_denominator = 1 + 2 * origin_inner_product + origin_norms_u * origin_norms_v
            origin_denominator = tf.maximum(origin_denominator, self.min_norm)

            origin_numerator = (1 + 2 * origin_inner_product + origin_norms_v) * vectors_u + (
                    1 - origin_norms_u) * vectors_v
            origin_results = tf.math.divide(origin_numerator, origin_denominator)
            return changed_results

        return changed_results

    # modified by lxy
    # func to compute mobius matrix multiply
    def mobius_matmul(self, vectors, matrix, bias=None):
        vectors = vectors + self.eps
        changed_matrix_ = torch.matmul(vectors, matrix) + self.eps
        changed_matrix_norm = util_norm(changed_matrix_)
        changed_vector_norm = util_norm(vectors)
        changed_result = 1. / np.sqrt(self.radius) * util_tanh(
            changed_matrix_norm / changed_vector_norm * util_atanh(
                np.sqrt(self.radius) * changed_vector_norm)) / changed_matrix_norm * changed_matrix_
        if bias is None:
            origin_result = self.hyperbolic_projection(changed_result)
        else:
            origin_result = self.hyperbolic_projection(self.mobius_addition(changed_result, bias))

        if util.DEBUG:
            origin_matrix_ = tf.matmul(vectors, matrix) + self.eps
            origin_matrix_norm = util_norm(origin_matrix_)
            origin_vector_norm = util_norm(vectors)
            origin_result = 1. / np.sqrt(self.radius) * util_tanh(
                origin_matrix_norm / origin_vector_norm * util_atanh(
                    np.sqrt(self.radius) * origin_vector_norm)) / origin_matrix_norm * origin_matrix_
            if bias is None:
                origin_result = self.hyperbolic_projection(origin_result)
            else:
                origin_result = self.hyperbolic_projection(self.mobius_addition(origin_result, bias))
            util.judge_change_equal(origin_result, changed_result)

    def log_map_zero(self, vectors):
        # vectors_norm = tf.maximum(tf_norm(vectors), self.min_norm)
        # vectors = vectors * 1. / np.sqrt(self.radius) * tf_atanh(np.sqrt(self.radius) * vectors_norm) / vectors_norm
        # return vectors
        diff = vectors + self.eps
        norm_diff = util_norm(diff)
        return 1.0 / np.sqrt(self.radius) * util_atanh(np.sqrt(self.radius) * norm_diff) / norm_diff * diff

    def exp_map_zero(self, vectors):
        # vectors_norm = tf.maximum(tf_norm(vectors), self.min_norm)
        diff = vectors + self.eps
        vectors_norm = util_norm(diff)
        vectors = util_tanh(np.sqrt(self.radius) * vectors_norm) * vectors / (np.sqrt(self.radius) * vectors_norm)
        return vectors

    # 这函数的作用不是很清楚
    def hyperbolic_projection(self, vectors):
        # Projection operation. Need to make sure hyperbolic embeddings are inside the unit ball.
        # vectors_norm = tf.maximum(tf_norm(vectors), self.min_norm)
        # max_norm = self.max_norm / np.sqrt(self.radius)
        # cond = tf.squeeze(vectors_norm > max_norm)
        # projected = vectors / vectors_norm * max_norm
        # return tf.where(cond, projected, vectors)
        return tf.clip_by_norm(t=vectors, clip_norm=self.max_norm / np.sqrt(self.radius), axes=[-1])

    def square_distance(self, u, v):
        distance = util_atanh(np.sqrt(self.radius) * util_norm(self.mobius_addition(-u, v)))
        distance = distance * 2 / np.sqrt(self.radius)
        return distance
