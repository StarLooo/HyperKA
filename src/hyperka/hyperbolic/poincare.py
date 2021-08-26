# -*- coding: utf-8 -*-
import numpy as np
import torch

from hyperka.hyperbolic.euclidean import EuclideanManifold
from hyperka.hyperbolic.util import util_norm, util_tanh, util_atanh

DEBUG = False


class PoincareManifold(EuclideanManifold):
    name = "poincare"

    def __init__(self, eps=1e-15, projection_eps=1e-5, radius=1.0, **kwargs):
        super(PoincareManifold, self).__init__(**kwargs)
        self.eps = eps
        self.projection_eps = projection_eps
        self.radius = radius  # the radius of the poincare ball
        self.max_norm = 1 - eps
        self.min_norm = eps

    def distance(self, u, v):
        sq_u_norm = torch.sum(u * u, dim=-1, keepdim=True)
        sq_v_norm = torch.sum(v * v, dim=-1, keepdim=True)
        sq_u_norm = torch.clip(sq_u_norm, min=0.0, max=self.max_norm)
        sq_v_norm = torch.clip(sq_v_norm, min=0.0, max=self.max_norm)
        sq_dist = torch.sum(torch.pow(u - v, 2), dim=-1, keepdim=True)
        distance = torch.acosh(
            1 + self.eps + (sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm)) * 2))
        return distance

    def mobius_addition(self, vectors_u, vectors_v):
        norms_u = self.radius * torch.sum(torch.square(vectors_u), dim=-1, keepdim=True)
        norms_v = self.radius * torch.sum(torch.square(vectors_v), dim=-1, keepdim=True)

        inner_product = self.radius * torch.sum(vectors_u * vectors_v, dim=-1, keepdim=True)
        denominator = 1 + 2 * inner_product + norms_u * norms_v

        denominator = torch.maximum(denominator, torch.full_like(denominator, self.min_norm))

        numerator = (1 + 2 * inner_product + norms_v) * vectors_u + (1 - norms_u) * vectors_v
        results = torch.divide(numerator, denominator)
        return results

    def mobius_matmul(self, vectors, matrix, bias=None):
        vectors = vectors + self.eps
        matrix_ = torch.matmul(vectors, matrix) + self.eps
        matrix_norm = util_norm(matrix_)
        vector_norm = util_norm(vectors)
        result = 1. / np.sqrt(self.radius) * util_tanh(
            matrix_norm / vector_norm * util_atanh(
                np.sqrt(self.radius) * vector_norm)) / matrix_norm * matrix_
        if bias is None:
            result = self.hyperbolic_projection(result)
        else:
            result = self.hyperbolic_projection(self.mobius_addition(result, bias))

        return result

    def log_map_zero(self, vectors):
        diff = vectors + self.eps
        norm_diff = util_norm(diff)
        return 1.0 / np.sqrt(self.radius) * util_atanh(np.sqrt(self.radius) * norm_diff) / norm_diff * diff

    def exp_map_zero(self, vectors):
        diff = vectors + self.eps
        vectors_norm = util_norm(diff)
        vectors = util_tanh(np.sqrt(self.radius) * vectors_norm) * vectors / (np.sqrt(self.radius) * vectors_norm)
        return vectors

    def hyperbolic_projection(self, vectors):
        # Projection operation. Need to make sure hyperbolic embeddings are inside the unit ball.

        # 由于没有在torch找到对应的轮子，所以自己造了一个
        vectors_norm = torch.maximum(util_norm(vectors), torch.full_like(vectors, self.min_norm))
        max_norm = self.max_norm / np.sqrt(self.radius)
        cond = torch.squeeze(vectors_norm < max_norm)
        projected = vectors / vectors_norm * max_norm
        return torch.where(condition=cond, input=vectors, other=projected)

    def square_distance(self, u, v):
        distance = util_atanh(np.sqrt(self.radius) * util_norm(self.mobius_addition(-u, v)))
        distance = distance * 2 / np.sqrt(self.radius)
        return distance
