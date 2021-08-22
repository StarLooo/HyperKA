# -*- coding: utf-8 -*-
import random

from hyperka.et_funcs.triples import Triples

DEBUG = False  # 本机测试时需要做一些调整


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


# 根据参数的要求读取相应的文件
def get_input(all_file, train_file, test_file, if_cross=False, ins_ids=None, onto_ids=None):
    # if_cross=True表示数据集中既有instance又有type
    if if_cross:
        assert ins_ids is not None and onto_ids is not None
        print("read entity types...")
        train_triples = read_triples(train_file)
        print("# all train entity types len:", len(train_triples))
        test_triples = read_triples(test_file)
        print("# all test entity types len:", len(test_triples))
        all_triples = train_triples | test_triples
        print("# all entity types len:", len(all_triples))

        # **************************************train**************************************
        train_heads_id_list = list()
        train_tails_id_list = list()
        train_ins_name_set = set()
        for triple in train_triples:
            h, t = triple[0], triple[2]
            # 若头实体不是已知的instance或者尾实体不是已知的type则跳过这一个三元组(该三元组不能用于训练)
            if h not in ins_ids.keys() or t not in onto_ids.keys():
                continue
            this_head_id, this_tail_id = ins_ids[h], onto_ids[t]
            # 头部instance如果之前已经出现了，也跳过
            # TODO:为什么这里也需要跳过？
            if h in train_ins_name_set:
                continue
            train_ins_name_set.add(h)  # 将已经出现过的头放入train_ins_name_set中
            train_heads_id_list.append(this_head_id)
            train_tails_id_list.append(this_tail_id)
            # 在训练集中train_heads_id_list[i]对应的实体一定属于train_tails_id_list[i]对应的这个类型
            assert len(train_heads_id_list) == len(train_tails_id_list)
        print("# selected train entity types len:", len(train_heads_id_list))

        # **************************************test**************************************
        test_head_tails_id_dict = dict()
        for triple in test_triples:
            h, t = triple[0], triple[2]
            # 若头实体不是已知的instance或者尾实体不是已知的type则跳过这一个三元组(该三元组不能用于测试)
            if h not in ins_ids.keys() or t not in onto_ids.keys():
                continue
            this_head_id, this_tail_id = ins_ids[h], onto_ids[t]
            if this_head_id not in test_head_tails_id_dict.keys():
                test_head_tails_id_dict[this_head_id] = set()
            test_head_tails_id_dict[this_head_id].add(this_tail_id)

        test_heads_id_list = list()
        test_tails_id_list = list()
        # TODO: test_head_tails_id_list在测试时的作用还不是很清楚
        test_head_tails_id_list = list()
        test_ins_name_set = set()
        for triple in test_triples:
            h, t = triple[0], triple[2]
            # 同上
            if h not in ins_ids.keys() or t not in onto_ids.keys():
                continue
            this_head_id, this_tail_id = ins_ids[h], onto_ids[t]
            # 同上
            if h in test_ins_name_set:
                continue
            # 过滤掉已经在训练集中的头部
            if h in train_ins_name_set:
                continue
            test_ins_name_set.add(h)  # 将已经出现过的头放入test_ins_name_set中
            test_heads_id_list.append(this_head_id)
            test_tails_id_list.append(this_tail_id)
            test_head_tails_id_list.append(list(test_head_tails_id_dict[this_head_id]))
            # 在测试集中test_heads_id_list[i]对应的实体一定属于test_tails_id_list[i]对应的这个类型
            assert len(test_heads_id_list) == len(test_tails_id_list)
        print("# selected test entity types len:", len(test_heads_id_list))
        # 注意返回的嵌套列表结构
        return [[train_heads_id_list, train_tails_id_list],
                [test_heads_id_list, test_tails_id_list, test_head_tails_id_list]]

    else:
        print("read KG triples...")
        if "insnet" in all_file:
            graph_name = "instance"  # KG中的所有节点表示的实体都是instance
        else:
            graph_name = "ontology"  # KG中的所有节点表示的实体都是type
        all_triples = read_triples(all_file, drop_rate=0.9)
        print("all triples length:", len(all_triples))

        # 获得实体set和关系set
        ents, rels = get_ents_and_rels(all_triples)
        # 为实体和关系分配id
        ent_ids, rel_ids = generate_mapping_id(all_triples, ents, rels)
        # 将三元组的三个部分id化
        ids_triples = get_ids_triples(all_triples, ent_ids, rel_ids)
        # 用ids_triples初始化自定义的Triples类，更好地管理所需要的三元组的各种变换形式，重新赋给triples
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

        # 注意返回值的结构
        return [triples, train_ids_triples, test_ids_triples, total_ents_num,
                total_rels_num, total_triples_num], ent_ids


# 从给定的文件中读取(h,r,t)三元组，返回给定文件中所有(h,r,t)三元组组成的set
def read_triples(file, drop_rate=0.0):
    assert 0.0 <= drop_rate < 1
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            if DEBUG and drop_rate > 0.0:
                rand_num = random.uniform(0.0, 1.0)
                if rand_num <= drop_rate:
                    continue
            params = line.strip('\n').split('\t')
            assert len(params) == 3
            h = params[0]
            r = params[1]
            t = params[2]
            triples.add((h, r, t))
        f.close()
    return triples


# 输入一个triples三元组set
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
        if DEBUG:
            if triple[0] not in ent_ids.keys() or triple[1] not in rel_ids.keys() or triple[2] not in ent_ids.keys():
                continue
        ids_triples.add((ent_ids[triple[0]], rel_ids[triple[1]], ent_ids[triple[2]]))
    print("ids_triples len:", len(ids_triples))
    return ids_triples


# 对ents表示的所有实体和rels表示的所有关系进行排序
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

    assert len(ent_ids) == len(set(ents)) and len(rel_ids) == len(set(rels))
    return ent_ids, rel_ids


# TODO:唯一的作用在test_funcs.py中的eval_type_hyperbolic()函数，但是test_funcs没怎么看到，不过好在不涉及tf代码，不需要修改
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
