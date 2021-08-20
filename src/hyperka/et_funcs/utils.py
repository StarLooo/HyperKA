# -*- coding: utf-8 -*-
import numpy as np
import time

from hyperka.et_funcs.triples import Triples


# 输入一个triples
# 输出其实体(包括头实体和尾实体)set和关系set
def get_ents_and_rels(triples):
    heads = set([triple[0] for triple in triples])
    tails = set([triple[2] for triple in triples])
    rels = set([triple[1] for triple in triples])
    ents = heads | tails
    return ents, rels


# 输入一个triples,记录实体id的ent_ids字典和记录关系id的rel_ids字典
# 输出三元组(head_id, rel_id, tail_id)组成的集合
def get_ids_triples(triples, ent_ids, rel_ids):
    ids_triples = set()
    for triple in triples:
        ids_triples.add((ent_ids[triple[0]], rel_ids[triple[1]], ent_ids[triple[2]]))
    return ids_triples


# 对elements_set中ents表示的所有实体和rels表示的所有关系进行排序
def sort_elements(triples, ents, rels):
    ents_cnt_dict = dict()
    rels_cnt_dict = dict()
    for h, r, t in triples:
        if h in ents:
            ents_cnt_dict[h] = ents_cnt_dict.get(h, 0) + 1
        if r in rels:
            rels_cnt_dict[r] = rels_cnt_dict.get(r, 0) + 1
        if t in ents:
            ents_cnt_dict[t] = ents_cnt_dict.get(t, 0) + 1

    sorted_ents_list = sorted(ents_cnt_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_ents_list = [x[0] for x in sorted_ents_list]

    sorted_rels_list = sorted(rels_cnt_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_rels_list = [x[0] for x in sorted_rels_list]

    return ordered_ents_list, ordered_rels_list


# 将triples中ents表示的所有实体和rels表示的所有关系映射为id，返回表示这种映射的dict
def generate_mapping_id(triples, ents, rels, ordered=True):
    ent_ids = dict()
    rel_ids = dict()

    if ordered:
        ordered_ents_list, ordered_rels_list = sort_elements(triples, ents, rels)
        for i in range(len(ordered_ents_list)):
            ent_ids[ordered_ents_list[i]] = i
        for i in range(len(ordered_rels_list)):
            rel_ids[ordered_rels_list[i]] = i
    else:
        ent_index_cnt = 0
        for ent in ents:
            if ent not in ent_ids:
                ent_ids[ent] = ent_index_cnt
                ent_index_cnt += 1

        rel_index_cnt = 0
        for rel in rels:
            if rel not in rel_ids:
                rel_ids[rel] = rel_index_cnt
                rel_index_cnt += 1

    assert len(ent_ids) == len(set(ents))
    return ent_ids, rel_ids


# 根据参数的要求读取相应的文件
# TODO: 部分代码没看懂
def get_input(all_file, train_file, test_file, if_cross=False, ins_ids=None, onto_ids=None):
    # if_cross=True表示既有instance又有type,用于类型推断任务
    if if_cross:
        print("read entity types...")
        train_triples = read_triples(train_file)
        print("# all train entity types len:", len(train_triples))
        test_triples = read_triples(test_file)
        print("# all test entity types len:", len(test_triples))
        all_triples = train_triples | test_triples
        print("# all entity types len:", len(all_triples))

        train_heads_id_list = list()
        train_tails_id_list = list()
        train_ins_name_set = set()
        for triple in train_triples:
            h, t = triple[0], triple[2]
            # filter the entities that not have triples in the KG
            # 若头实体不是已知的instance或者尾实体不是已知的type则跳过这一个三元组(该三元组不能用于训练)
            if h not in ins_ids.keys() or t not in onto_ids.keys():
                continue
            this_head_id, this_tail_id = ins_ids[h], onto_ids[t]
            # 头部instance如果重复出现了，也跳过
            # TODO:为什么这里需要跳过？
            if h in train_ins_name_set:
                continue
            train_ins_name_set.add(h)
            train_heads_id_list.append(this_head_id)
            train_tails_id_list.append(this_tail_id)
            # 在训练集中train_heads_id_list[i]对应的实体一定属于train_tails_id_list[i]对应的这个类型
            assert len(train_heads_id_list) == len(train_tails_id_list)
        print("# selected train entity types len:", len(train_heads_id_list))

        test_head_tails_id_dict = dict()
        for triple in test_triples:
            h, t = triple[0], triple[2]
            # filter the entities that not have triples in the KG
            # 若头实体不是已知的instance或者尾实体不是已知的type则跳过这一个三元组(该三元组不能用于测试)
            if h not in ins_ids.keys() or t not in onto_ids.keys():
                continue
            this_head_id, this_tail_id = ins_ids[h], onto_ids[t]
            if this_head_id not in test_head_tails_id_dict.keys():
                test_head_tails_id_dict[this_head_id] = set()
            test_head_tails_id_dict[this_head_id].add(this_tail_id)

        # ***************************************
        test_heads_id_list = list()
        test_tails_id_list = list()
        # TODO: test_head_tails_id_list的作用还不是很清楚
        test_head_tails_id_list = list()
        test_ins_name_set = set()
        for triple in test_triples:
            h, t = triple[0], triple[2]
            # filter the entities that not have triples in the KG
            # 同上
            if h not in ins_ids.keys() or t not in onto_ids.keys():
                continue
            this_head_id, this_tail_id = ins_ids[h], onto_ids[t]
            # 同上
            if h in test_ins_name_set:
                continue
            # filter the instances in training data
            if h in train_ins_name_set:
                continue
            test_ins_name_set.add(h)
            test_heads_id_list.append(this_head_id)
            test_tails_id_list.append(this_tail_id)
            test_head_tails_id_list.append(list(test_head_tails_id_dict[this_head_id]))
            # 在测试集中train_heads_id_list[i]对应的实体一定属于train_tails_id_list[i]对应的这个类型
            assert len(test_heads_id_list) == len(test_tails_id_list)
        print("# selected test entity types len:", len(test_heads_id_list))

        return [[train_heads_id_list, train_tails_id_list],
                [test_heads_id_list, test_tails_id_list, test_head_tails_id_list]]

    else:
        print("read KG triples...")
        if "insnet" in all_file:
            graph_name = "instance"  # 图中的所有节点表示的实体都是instance
        else:
            graph_name = "ontology"  # 图中的所有节点表示的实体都是type

        all_triples = read_triples(all_file)
        print("all triples length:", len(all_triples))

        # 将实体、关系以及三元组全部id化
        ents, rels = get_ents_and_rels(all_triples)
        # 为实体和关系分配id
        ent_ids, rel_ids = generate_mapping_id(all_triples, ents, rels)
        # 将三元组的三个部分id化
        ids_triples = get_ids_triples(all_triples, ent_ids, rel_ids)
        # 用ids_triples初始化自定义的Triples类，更好地管理所需要的三元组，重新赋给triples
        triples = Triples(ids_triples)

        total_ents_num = len(triples.ents)
        total_rels_num = len(triples.rels)
        total_triples_num = len(triples.triple_list)
        print("total " + graph_name + " ents num:", total_ents_num)
        print("total " + graph_name + " rels num:", total_rels_num)
        print("total " + graph_name + " triples num:", total_triples_num)

        train_triples = read_triples(train_file)
        train_ids_triples = get_ids_triples(train_triples, ent_ids, rel_ids)

        test_triples = read_triples(test_file)
        test_ids_triples = get_ids_triples(test_triples, ent_ids, rel_ids)

        return [triples, train_ids_triples, test_ids_triples, total_ents_num,
                total_rels_num, total_triples_num], ent_ids


# 读入全部所需数据
def read_input(folder):
    # 使用DB111K-174数据集
    if "yago" not in folder:
        # insnet 实例和实例的关系
        insnet, ins_ids = get_input(all_file=folder + "db_insnet.txt",
                                    train_file=folder + "db_insnet_train.txt",
                                    test_file=folder + "db_insnet_test.txt")
        # onto 类型和类型的关系
        onto, onto_ids = get_input(all_file=folder + "db_onto_small_mini.txt",
                                   train_file=folder + "db_onto_small_train.txt",
                                   test_file=folder + "db_onto_small_test.txt")
        # instype 实例和类型的关系
        instype = get_input(all_file=folder + "db_InsType_mini.txt",
                            train_file=folder + "db_InsType_train.txt",
                            test_file=folder + "db_InsType_test.txt",
                            if_cross=True,
                            ins_ids=ins_ids,
                            onto_ids=onto_ids)
    # 使用YAGO26K-906数据集
    else:
        # insnet 实例和实例的关系
        insnet, ins_ids = get_input(all_file=folder + "yago_insnet_mini.txt",
                                    train_file=folder + "yago_insnet_train.txt",
                                    test_file=folder + "yago_insnet_test.txt")
        # onto 类型和类型的关系
        onto, onto_ids = get_input(all_file=folder + "yago_ontonet.txt",
                                   train_file=folder + "yago_ontonet_train.txt",
                                   test_file=folder + "yago_ontonet_test.txt")
        # instype 实例和类型的关系
        instype = get_input(all_file=folder + "yago_InsType_mini.txt",
                            train_file=folder + "yago_InsType_train.txt",
                            test_file=folder + "yago_InsType_test.txt",
                            if_cross=True,
                            ins_ids=ins_ids,
                            onto_ids=onto_ids)

    # 注意insnet, onto, instype都是list，其内部每个元素含义及其具体结构详见get_input()函数内的注释
    return insnet, onto, instype


# 将pairs中的二元组的信息写入file指定的文件中
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
            h = params[0]
            r = params[1]
            t = params[2]
            triples.add((h, r, t))
        f.close()
    return triples


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
        # attrs_embeds = np.sum(attr_embeddings[attrs_index,], axis=0)
        attrs_embeds = np.sum(attr_embeddings[attrs_index], axis=0)
        if ent_attrs_embeddings is None:
            ent_attrs_embeddings = attrs_embeds
        else:
            ent_attrs_embeddings = np.row_stack((ent_attrs_embeddings, attrs_embeds))
    print("shape of ent_attr_embeds: {}".format(ent_attrs_embeddings.shape))
    print("generating ent features costs: {:.3f} s".format(time.time() - t1))
    return ent_attrs_embeddings
