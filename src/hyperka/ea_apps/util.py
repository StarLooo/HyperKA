# -*- coding: utf-8 -*-
import os

import numpy as np
import scipy.sparse as sp
import time
import math
import torch
import torch.nn as nn


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# 初始化嵌入向量
# TODO: 注意tf版代码在这里用了系数为0.01的L2正则化(可能需要实验验证是否需要)，而torch的L2正则应该是在optimizer中实现的
def embed_init(size, name, method='xavier_uniform', data_type=torch.float64):
    # Xavier均匀分布(default)
    if method == 'xavier_uniform':
        print("init embeddings using", "xavier_uniform", "with size of", size)
        embeddings = nn.init.xavier_uniform_(
            tensor=torch.empty(size=size, dtype=data_type, requires_grad=True, device=try_gpu()))

    # 截断正态分布
    elif method == 'truncated_normal':
        print("init embeddings using", "truncated_normal", "with size of", size)
        embeddings = nn.init.trunc_normal_(
            tensor=torch.empty(size=size, dtype=data_type, requires_grad=True, device=try_gpu()), mean=0,
            std=1.0 / math.sqrt(size[1]))

    # 均匀分布
    else:
        print("init embeddings using", "random_uniform", "with size of", size)
        embeddings = nn.init.uniform_(
            tensor=torch.empty(size=size, dtype=data_type, requires_grad=True, device=try_gpu()), a=-0.001, b=0.001)

    return embeddings


# 将稀疏表示的adjacent_graph转换为tuple 表示
def sparse_to_tuple(sparse_matrix):
    def to_tuple(matrix):
        if not sp.isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data
        shape = matrix.shape
        # coords中的tuple记录顶点之间的邻接关系(有多少条边就有多少个这样的tuple)
        # values记录对应的边上的权重
        # shape记录图的形状
        return coords, values, shape

    if isinstance(sparse_matrix, list):
        for i in range(len(sparse_matrix)):
            sparse_matrix[i] = to_tuple(sparse_matrix[i])
    else:
        return to_tuple(sparse_matrix)


# 对adjacent_graph进行对称正则化处理
# TODO: 对称正则化处理的作用
def normalize_adj(adjacent_graph):
    """Symmetrically normalize adjacency matrix."""
    adjacent_graph = sp.coo_matrix(adjacent_graph)
    rowsum = np.array(adjacent_graph.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adjacent_graph.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# 对adjacent_graph进行预处理,将其正则化并转化为tuple的表示
def preprocess_adjacent_graph(adjacent_graph):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # TODO: 为什么这里要加上一个单位阵
    processed_adjacent_graph = normalize_adj(adjacent_graph + sp.eye(adjacent_graph.shape[0]))
    # processed_adjacent_graph = adjacent_graph  # GAT暂时先不做处理
    return sparse_to_tuple(processed_adjacent_graph)


# 根据triples生成对应的无权无向图,generate_graph的特例
def generate_no_weighted_undirected_adjacent_graph(total_ent_num, triples):
    start = time.time()
    edges = dict()
    for tripe in triples:
        if tripe[0] not in edges.keys():
            edges[tripe[0]] = set()
        if tripe[2] not in edges.keys():
            edges[tripe[2]] = set()
        edges[tripe[0]].add(tripe[2])
        edges[tripe[2]].add(tripe[0])

    # 用sp.coo_matrix()函数稀疏化表示
    row = list()
    col = list()
    for i in range(total_ent_num):
        # 表示id为i的实体并不在该KG对应的无向图中
        if i not in edges.keys():
            continue
        key = i
        values = edges[key]
        add_key_len = len(values)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(values))
    data_len = len(row)
    data = np.ones(data_len)

    # graph = torch.sparse_coo_tensor(indices=[row, col], values=data, size=(total_ent_num, total_ent_num))
    # print("graph:", graph)
    # print(graph.is_coalesced())
    # graph = graph.coalesce()
    # print("graph:", graph)
    # os.system("pause")

    # 进行稀疏化表示
    adjacent_graph = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    # 经过preprocess_adjacent_graph()后adjacent_graph已经是用tuple表示的了
    adjacent_graph = preprocess_adjacent_graph(adjacent_graph)

    end = time.time()
    print('generating KG costs time: {:.4f}s'.format(end - start))
    return adjacent_graph


# 根据triples生成对应的图(默认为无向无权图)
def generate_graph(total_ents_num, total_rels_num, triples):
    start = time.time()
    near_ents = dict()  # 各实体的邻居实体
    near_rels = dict()  # 各实体的邻居关系
    ents_near_ents_num = torch.zeros(size=(total_ents_num,), dtype=torch.int, device=try_gpu(),
                                     requires_grad=False)  # 各实体的邻居实体数
    ents_near_rels_num = torch.zeros(size=(total_ents_num,), dtype=torch.int, device=try_gpu(),
                                     requires_grad=False)  # 各实体的邻居关系数
    rels_near_ents_num = torch.zeros(size=(total_rels_num,), dtype=torch.int, device=try_gpu(),
                                     requires_grad=False)  # 各关系的邻居实体数

    for tripe in triples:
        h, r, t = tripe
        # near_ents:
        if h not in near_ents.keys():
            near_ents[h] = set()
        if t not in near_ents.keys():
            near_ents[t] = set()
        near_ents[h].add(t)
        near_ents[t].add(h)

        # near_rels:
        if h not in near_rels.keys():
            near_rels[h] = set()
        if t not in near_rels.keys():
            near_rels[t] = set()
        near_rels[h].add(r)
        near_rels[t].add(r)

    near_ents_rows_list, near_ents_cols_list, near_ents_values_list = list(), list(), list()
    near_rels_rows_list, near_rels_cols_list, near_rels_values_list = list(), list(), list()

    for ent_id in range(total_ents_num):
        if ent_id not in near_ents.keys():
            continue
        source = ent_id
        near_ents_of_source = list(near_ents[source])
        ents_near_ents_num[ent_id] = len(near_ents_of_source)
        near_ents_values = np.ones(len(near_ents_of_source), dtype=int).tolist()
        near_ents_rows_list.extend((source * np.ones(len(near_ents_of_source), dtype=int)).tolist())
        near_ents_cols_list.extend(near_ents_of_source)
        near_ents_values_list.extend(near_ents_values)

        near_rels_of_source = list(near_rels[source])
        ents_near_rels_num[ent_id] = len(near_rels_of_source)
        rels_near_ents_num[near_rels_of_source] += 1
        near_rels_values = near_rels_of_source
        near_rels_rows_list.extend((source * np.ones(len(near_rels_of_source), dtype=int)).tolist())
        near_rels_cols_list.extend(near_rels_of_source)
        near_rels_values_list.extend(near_rels_values)

    # 进行稀疏化表示
    near_ents_adj = torch.sparse_coo_tensor(indices=[near_ents_rows_list, near_ents_cols_list],
                                            values=near_ents_values_list, size=(total_ents_num, total_ents_num),
                                            device=try_gpu(), requires_grad=False).coalesce()
    near_rels_adj = torch.sparse_coo_tensor(indices=[near_rels_rows_list, near_rels_cols_list],
                                            values=near_rels_values_list, size=(total_ents_num, total_rels_num),
                                            device=try_gpu(), requires_grad=False).coalesce()

    near_ents_graph = (near_ents_adj, ents_near_ents_num)
    near_rels_graph = (near_rels_adj, ents_near_rels_num, rels_near_ents_num)

    # print("near_ents_adj:", near_ents_adj)
    # print("ents_near_ents_num:", ents_near_ents_num, ents_near_ents_num.shape, (ents_near_ents_num == 0).sum())
    # print("near_rels_adj:", near_rels_adj)
    # print("ents_near_rels_num:", ents_near_rels_num, ents_near_rels_num.shape, (ents_near_rels_num == 0).sum())
    # print("rels_near_ents_num:", rels_near_ents_num, rels_near_ents_num.shape, (rels_near_ents_num == 0).sum())
    # os.system("pause")

    end = time.time()
    print('generating KG costs time: {:.4f}s'.format(end - start))

    return near_ents_graph, near_rels_graph


# 根据triples生成对应的邻接图
def generate_adjacent_graph(total_ents_num, total_rels_num, triples):
    # return generate_no_weighted_undirected_adjacent_graph(total_ents_num, triples)
    return generate_graph(total_ents_num, total_rels_num, triples)

# These may be useless:
# def generate_adj_dict(total_e_num, triples):
#     one_adj = generate_adjacent_graph(total_e_num, triples)
#     adj = one_adj
#     x = adj[0].shape[0]
#     weighted_edges = dict()
#     mat = adj[0]
#     weight_mat = adj[1]
#     for i in range(x):
#         node1 = mat[i, 0]
#         node2 = mat[i, 1]
#         weight = weight_mat[i]
#         edges = weighted_edges.get(node1, set())
#         edges.add((node1, node2, weight))
#         weighted_edges[node1] = edges
#     assert len(weighted_edges) == adj[2][0]
#     return weighted_edges
#
# def uniform(shape, scale=0.05, name=None):
# """Uniform init."""
#     initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
#     return tf.Variable(initial, name=name)
#
#
# def glorot(shape, name=None):
#     """Glorot & Bengio (AISTATS 2010) init."""
#     init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
#     initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float64)
#     return tf.Variable(initial, name=name)
#
#
# def zeros(shape, name=None):
#     """All zeros."""
#     initial = tf.zeros(shape, dtype=tf.float64)
#     return tf.Variable(initial, name=name)
#
# def ones(shape, name=None):
#     """All ones."""
#     initial = tf.ones(shape, dtype=tf.float64)
#     return tf.Variable(initial, name=name)
