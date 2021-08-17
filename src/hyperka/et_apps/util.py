import numpy as np
import scipy.sparse as sp
import time
import math
import torch
import torch.nn as nn


# import tensorflow as tf

# 初始化嵌入向量
def embed_init(mat_x, mat_y, name, method='glorot_uniform_initializer', data_type=torch.float64):
    # Xavier均匀分布
    if method == 'glorot_uniform_initializer':
        print("init embeddings using", "glorot_uniform_initializer", "with dim of", mat_x, mat_y)
        embeddings = nn.init.xavier_uniform_(
            tensor=torch.empty(size=(mat_x, mat_y), dtype=data_type, requires_grad=True))
        # embeddings = tf.get_variable(name, shape=[mat_x, mat_y], initializer=tf.glorot_uniform_initializer(),
        #                              dtype=data_type)

    # 截断正态分布
    elif method == 'truncated_normal':
        print("init embeddings using", "truncated_normal", "with dim of", mat_x, mat_y)
        embeddings = nn.init.trunc_normal_(tensor=torch.empty(size=(mat_x, mat_y), dtype=data_type, requires_grad=True),
                                           std=1.0 / math.sqrt(mat_y))
        # embeddings = tf.Variable(tf.truncated_normal([mat_x, mat_y], stddev=1.0 / math.sqrt(mat_y), dtype=data_type),
        #                          name=name, dtype=data_type)

    # 均匀分布
    else:
        print("init embeddings using", "random_uniform", "with dim of", mat_x, mat_y)
        embeddings = nn.init.uniform_(tensor=torch.empty(size=(mat_x, mat_y), dtype=data_type, requires_grad=True),
                                      a=-0.001, b=0.001)
        # embeddings = tf.get_variable(name=name, dtype=data_type,
        #                              initializer=tf.random_uniform([mat_x, mat_y],
        #                                                            minval=-0.001, maxval=0.001, dtype=data_type))

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
    normalized_adjacent_graph = normalize_adj(adjacent_graph + sp.eye(adjacent_graph.shape[0]))
    return sparse_to_tuple(normalized_adjacent_graph)


# 根据triples生成对应的无权无向图
# TODO:该函数内部具体如何对图进行稀疏表示以及预处理的部分目前被当成黑箱处理,未来可以仔细研究一下
def generate_no_weighted_adjacent_graph(total_ent_num, triples):
    start = time.time()
    # 用邻接表表示图
    # edges中的key是某实体h的id,对应的value是与之相关(关系用r表示)的一系列实体t的id的set,其中(h,r,t)构成一个triples中的三元组
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
        # 表示id为i的实体不属于该KG
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

    adjacent_graph = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    # 经过preprocess_adjacent_graph后adjacent_graph已经是用tuple表示的了
    adjacent_graph = preprocess_adjacent_graph(adjacent_graph)

    end = time.time()
    print('generating KG costs time: {:.4f}s'.format(end - start))
    return adjacent_graph


# 根据triples生成对应的邻接图
def generate_adjacent_graph(total_ents_num, triples):
    return generate_no_weighted_adjacent_graph(total_ents_num, triples)


def generate_adjacent_dict(total_e_num, triples):
    one_adj = generate_no_weighted_adjacent_graph(total_e_num, triples)
    adj = one_adj
    x = adj[0].shape[0]
    weighted_edges = dict()
    mat = adj[0]
    weight_mat = adj[1]
    for i in range(x):
        node1 = mat[i, 0]
        node2 = mat[i, 1]
        weight = weight_mat[i]
        edges = weighted_edges.get(node1, set())
        edges.add((node1, node2, weight))
        weighted_edges[node1] = edges
    assert len(weighted_edges) == adj[2][0]
    return weighted_edges


# 这几个函数貌似没有用
# def uniform(shape, scale=0.05, name=None):
#     """Uniform init."""
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
#
# def ones(shape, name=None):
#     """All ones."""
#     initial = tf.ones(shape, dtype=tf.float64)
#     return tf.Variable(initial, name=name)

if __name__ == '__main__':
    # test sparse_to_tuple
    matrix = [[0, 4, 0, 3], [4, 0, 1, 0], [0, 1, 0, 7], [3, 0, 7, 0]]
    sparse_matrix = sp.coo_matrix(matrix)
    res = sparse_to_tuple(sparse_matrix)
    print(res[0])
    print(res[1])
    print(res[2])
