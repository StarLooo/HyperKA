# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import scipy.sparse as sp
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
# preprocess_adjacent_graph的测试可以看看下方该.py文件的main函数
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
# TODO:还有遗留的问题没解决，先用generate_no_weighted_undirected_adjacent_graph
def generate_graph(total_ents_num, total_rels_num, triples):
    start = time.time()
    near_ents = dict()  # 各实体的邻居实体
    near_rels = dict()  # 各实体的邻居关系
    near_ents_num = torch.empty(size=(total_ents_num,), dtype=torch.int, device=try_gpu(),
                                requires_grad=False)  # 各实体的邻居实体数
    near_rels_num = torch.empty(size=(total_ents_num,), dtype=torch.int, device=try_gpu(),
                                requires_grad=False)  # 各实体的邻居关系数
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
    assert len(near_ents.keys()) == len(near_ents.keys()) == total_ents_num

    near_ents_rows_list, near_ents_cols_list, near_ents_values_list = list(), list(), list()
    near_rels_rows_list, near_rels_cols_list, near_rels_values_list = list(), list(), list()

    for ent_id in range(total_ents_num):
        if ent_id not in near_ents.keys():
            continue
        source = ent_id
        near_ents_of_source = list(near_ents[source])
        near_ents_num[ent_id] = len(near_ents_of_source)
        near_ents_values = np.ones(len(near_ents_of_source), dtype=int).tolist()
        near_ents_rows_list.extend((source * np.ones(len(near_ents_of_source), dtype=int)).tolist())
        near_ents_cols_list.extend(near_ents_of_source)
        near_ents_values_list.extend(near_ents_values)

        near_rels_of_source = list(near_rels[source])
        near_rels_num[ent_id] = len(near_rels_of_source)
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

    near_ents_graph = (near_ents_adj, near_ents_num)
    near_rels_graph = (near_rels_adj, near_rels_num)

    # print("near_ents_adj:", near_ents_adj)
    # print("near_ents_num:", near_ents_num, near_ents_num.shape)
    # print("near_rels_adj:", near_rels_adj)
    # print("near_rels_num:", near_rels_num, near_rels_num.shape)
    # os.system("pause")

    end = time.time()
    print('generating KG costs time: {:.4f}s'.format(end - start))

    return near_ents_graph, near_rels_graph


# 根据triples生成对应的邻接图
def generate_adjacent_graph(total_ents_num, total_rels_num, triples):
    # return generate_no_weighted_undirected_adjacent_graph(total_ents_num, triples)
    return generate_graph(total_ents_num, total_rels_num, triples)


if __name__ == '__main__':
    # test sparse_to_tuple
    matrix = [[0, 4, 0, 3], [4, 0, 1, 0], [0, 1, 0, 7], [3, 0, 7, 0]]
    sparse_matrix = sp.coo_matrix(matrix)
    res = sparse_to_tuple(sparse_matrix)
    print(res[0])
    print(res[1])
    print(res[2])
