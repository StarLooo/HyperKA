import numpy as np
import time

from hyperka.ea_funcs.triples import Triples


# TODO: may be useless
# 读入非dbp15k数据集中全部所需数据(可能是作者还用了dbp15k以外的数据集来进行实验)
def read_other_input(folder):
    source_triples = Triples(read_triples(folder + 'triples_1'))
    target_triples = Triples(read_triples(folder + 'triples_2'))

    total_ents_num = len(source_triples.ents | target_triples.ents)
    total_rels_num = len(source_triples.rels | target_triples.rels)
    total_triples_num = len(source_triples.triple_list) + len(target_triples.triple_list)
    print('total ents num:', total_ents_num)
    print('total rels num:', len(source_triples.rels), len(target_triples.rels), total_rels_num)
    print('total triples num: %d + %d = %d' % (
        len(source_triples.triples), len(target_triples.triples), total_triples_num))

    ref_source_aligned_ents, ref_target_aligned_ents = read_aligned_pairs(folder + 'ref_ent_ids')
    print("aligned entities in train file:", len(ref_source_aligned_ents))

    sup_source_aligned_ents, sup_target_aligned_ents = read_aligned_pairs(folder + 'sup_ent_ids')

    return \
        source_triples, target_triples, sup_source_aligned_ents, sup_target_aligned_ents, \
        ref_source_aligned_ents, ref_target_aligned_ents, total_ents_num, total_rels_num


# 读入dbp15k数据集中全部所需数据
def read_dbp15k_input(folder):
    # 读取源知识图谱(source_KG)中的三元组并用于初始化Triples类，便于管理
    source_triples = Triples(read_triples(folder + 'triples_1'))
    # 读取目标知识图谱(target_KG)中的三元组并用于初始化Triples类，便于管理
    target_triples = Triples(read_triples(folder + 'triples_2'))

    total_ents_num = len(source_triples.ents | target_triples.ents)
    total_rels_num = len(source_triples.rels | target_triples.rels)
    total_triples_num = len(source_triples.triple_list) + len(target_triples.triple_list)
    print('total ents num:', total_ents_num)
    print('total rels num:', len(source_triples.rels), len(target_triples.rels), total_rels_num)
    print('total triples num: %d + %d = %d' % (
        len(source_triples.triples), len(target_triples.triples), total_triples_num))

    # TODO: 不清楚这里路径中mtranse的含义是不是与MTransE算法有关，还是仅仅使用了相同的数据集而已
    # 事实上"mtranse"确实在我们使用的dbk15数据集的路径中
    if 'mtranse' in folder:
        ref_source_aligned_ents, ref_target_aligned_ents = read_aligned_pairs(folder + 'ref_pairs')
    else:
        ref_source_aligned_ents, ref_target_aligned_ents = read_aligned_pairs(folder + 'ref_ent_ids')
    print("aligned entities in train file:", len(ref_source_aligned_ents))

    if 'mtranse' in folder:
        sup_source_aligned_ents, sup_target_aligned_ents = read_aligned_pairs(folder + 'sup_pairs')
    else:
        sup_source_aligned_ents, sup_target_aligned_ents = read_aligned_pairs(folder + 'sup_ent_ids')

    # 注意返回值的结构和顺序
    return \
        source_triples, target_triples, sup_source_aligned_ents, sup_target_aligned_ents, \
        ref_source_aligned_ents, ref_target_aligned_ents, total_ents_num, total_rels_num


def generate_sup_triples(triples1, triples2, ents1, ents2):
    def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
        newly_triples = set()
        for r, t in rt_dict1.get(ent1, set()):
            newly_triples.add((ent2, r, t))
        for h, r in hr_dict1.get(ent1, set()):
            newly_triples.add((h, r, ent2))
        return newly_triples

    assert len(ents1) == len(ents2)
    newly_triples1, newly_triples2 = set(), set()
    for i in range(len(ents1)):
        newly_triples1 |= (generate_newly_triples(ents1[i], ents2[i], triples1.rt_dict, triples1.hr_dict))
        newly_triples2 |= (generate_newly_triples(ents2[i], ents1[i], triples2.rt_dict, triples2.hr_dict))
    print("supervised triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def add_sup_triples(triples1, triples2, sup_ent1, sup_ent2):
    newly_triples1, newly_triples2 = generate_sup_triples(triples1, triples2, sup_ent1, sup_ent2)
    triples1 = Triples(triples1.triples | newly_triples1, ori_triples=triples1.triples)
    triples2 = Triples(triples2.triples | newly_triples2, ori_triples=triples2.triples)
    print("now triples: {}, {}".format(len(triples1.triples), len(triples2.triples)))
    return triples1, triples2


def pair2file(file, pairs):
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


# 从给定的文件中读取(h,r,t)三元组，返回给定文件中所有(h,r,t)三元组组成的set
def read_triples(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 3
            h = int(params[0])
            r = int(params[1])
            t = int(params[2])
            triples.add((h, r, t))
        f.close()
    return triples


# 读取已经对齐的实体对
def read_aligned_pairs(file):
    source_aligned_ents, target_aligned_ents = list(), list()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            ent_1 = int(params[0])
            ent_2 = int(params[1])
            source_aligned_ents.append(ent_1)
            target_aligned_ents.append(ent_2)
        f.close()
        assert len(source_aligned_ents) == len(target_aligned_ents)
    return source_aligned_ents, target_aligned_ents


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def triples2ht_set(triples):
    ht_set = set()
    for h, r, t in triples:
        ht_set.add((h, t))
    print("the number of ht: {}".format(len(ht_set)))
    return ht_set


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def generate_adjacency_mat(triples1, triples2, ent_num, sup_ents):
    adj_mat = np.mat(np.zeros((ent_num, len(sup_ents)), dtype=np.int32))
    ht_set = triples2ht_set(triples1) | triples2ht_set(triples2)
    for i in range(ent_num):
        for j in sup_ents:
            if (i, j) in ht_set:
                adj_mat[i, sup_ents.index(j)] = 1
    print("shape of adj_mat: {}".format(adj_mat.shape))
    print("the number of 1 in adjacency matrix: {}".format(np.count_nonzero(adj_mat)))
    return adj_mat


def generate_adj_input_mat(adj_mat, d):
    w = np.random.randn(adj_mat.shape[1], d)
    m = np.matmul(adj_mat, w)
    print("shape of input adj_mat: {}".format(m.shape))
    return m


def generate_ent_attrs_sum(ent_num, ent_attrs1, ent_attrs2, attr_embeddings):
    t1 = time.time()
    ent_attrs_embeddings = None
    for i in range(ent_num):
        attrs_index = list(ent_attrs1.get(i, set()) | ent_attrs2.get(i, set()))
        assert len(attrs_index) > 0
        attrs_embeds = np.sum(attr_embeddings[attrs_index, :], axis=0)
        if ent_attrs_embeddings is None:
            ent_attrs_embeddings = attrs_embeds
        else:
            ent_attrs_embeddings = np.row_stack((ent_attrs_embeddings, attrs_embeds))
    print("shape of ent_attr_embeds: {}".format(ent_attrs_embeddings.shape))
    print("generating ent features costs: {:.3f} s".format(time.time() - t1))
    return ent_attrs_embeddings
