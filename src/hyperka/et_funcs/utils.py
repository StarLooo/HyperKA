# -*- coding: utf-8 -*-
from src.hyperka.et_funcs.triples import Triples


# 读入全部所需数据
def read_input(folder):
    # 使用DB111K-174数据集
    if "yago" not in folder:
        # insnet 实例和实例的关系
        insnet, ins_ids_dict = get_input(all_file=folder + "db_insnet.txt",
                                         train_file=folder + "db_insnet_train.txt",
                                         test_file=folder + "db_insnet_test.txt")
        # onto 类型和类型的关系
        onto, onto_ids_dict = get_input(all_file=folder + "db_onto_small_mini.txt",
                                        train_file=folder + "db_onto_small_train.txt",
                                        test_file=folder + "db_onto_small_test.txt")
        # instype 实例和类型的关系
        instype = get_input(all_file=folder + "db_InsType_mini.txt",
                            train_file=folder + "db_InsType_train.txt",
                            test_file=folder + "db_InsType_test.txt",
                            if_cross=True,
                            ins_ids_dict=ins_ids_dict,
                            onto_ids_dict=onto_ids_dict)

    # 使用YAGO26K-906数据集
    else:
        # insnet 实例和实例的关系
        insnet, ins_ids_dict = get_input(all_file=folder + "yago_insnet_mini.txt",
                                         train_file=folder + "yago_insnet_train.txt",
                                         test_file=folder + "yago_insnet_test.txt")
        # onto 类型和类型的关系
        onto, onto_ids_dict = get_input(all_file=folder + "yago_ontonet.txt",
                                        train_file=folder + "yago_ontonet_train.txt",
                                        test_file=folder + "yago_ontonet_test.txt")
        # instype 实例和类型的关系
        instype = get_input(all_file=folder + "yago_InsType_mini.txt",
                            train_file=folder + "yago_InsType_train.txt",
                            test_file=folder + "yago_InsType_test.txt",
                            if_cross=True,
                            ins_ids_dict=ins_ids_dict,
                            onto_ids_dict=onto_ids_dict)

    # 注意insnet, onto, instype都是list，其内部每个元素含义及其具体结构详见get_input()函数内的注释
    return insnet, onto, instype


# 根据参数的要求读取相应的文件
def get_input(all_file, train_file, test_file, if_cross=False, ins_ids_dict=None, onto_ids_dict=None):
    # if_cross=True表示数据集中既有instance又有type
    if if_cross:
        assert ins_ids_dict is not None and onto_ids_dict is not None

        print("read entity types...")
        train_triples_set = read_triples(train_file)
        print("# all train entity types len:", len(train_triples_set))
        test_triples_set = read_triples(test_file)
        print("# all test entity types len:", len(test_triples_set))
        all_triples_set = train_triples_set | test_triples_set
        print("# all entity types len:", len(all_triples_set))

        # **************************************train**************************************
        # train_heads_ids_list和train_tails_ids_list表示原始数据的所有三元组中去掉多余的三元组所剩下的(也就是实际选用的)
        # 三元组的所有头部的编号列表(train_heads_ids_list)和尾部的编号列表(train_tails_ids_list)，二者一一对应
        train_heads_ids_list = list()
        train_tails_ids_list = list()
        train_ins_names_set = set()
        for triple in train_triples_set:
            h, t = triple[0], triple[2]
            # 若头实体不是已知的instance或者尾实体不是已知的type则跳过这一个三元组(该三元组不能用于训练)
            if h not in ins_ids_dict.keys() or t not in onto_ids_dict.keys():
                continue
            this_head_id, this_tail_id = ins_ids_dict[h], onto_ids_dict[t]
            # 头部instance如果之前已经出现了，也跳过
            # TODO:为什么这里也需要跳过？
            if h in train_ins_names_set:
                continue
            train_ins_names_set.add(h)  # 将已经出现过的头放入train_ins_names_set中
            train_heads_ids_list.append(this_head_id)
            train_tails_ids_list.append(this_tail_id)
            # 在训练集中train_heads_ids_list[i]对应的实体一定属于train_tails_ids_list[i]对应的这个类型
            assert len(train_heads_ids_list) == len(train_tails_ids_list)
        print("# selected train entity types len:", len(train_heads_ids_list))

        # **************************************test**************************************
        # TODO: test_head_tails_id_dict在测试时的作用还不是很清楚
        test_head_tails_ids_dict = dict()
        for triple in test_triples_set:
            h, t = triple[0], triple[2]
            # 若头实体不是已知的instance或者尾实体不是已知的type则跳过这一个三元组(该三元组不能用于测试)
            if h not in ins_ids_dict.keys() or t not in onto_ids_dict.keys():
                continue
            this_head_id, this_tail_id = ins_ids_dict[h], onto_ids_dict[t]
            if this_head_id not in test_head_tails_ids_dict.keys():
                test_head_tails_ids_dict[this_head_id] = set()
            test_head_tails_ids_dict[this_head_id].add(this_tail_id)
        # test_heads_ids_list和test_tails_ids_list表示原始数据的所有三元组中去掉多余的三元组所剩下的(也就是实际选用的)
        # 三元组的所有头部的编号列表(test_heads_ids_list)和尾部的编号列表(test_tails_ids_list)，二者一一对应
        test_heads_ids_list = list()
        test_tails_ids_list = list()
        # TODO: test_head_tails_id_list在测试时的作用还不是很清楚
        test_head_tails_ids_list = list()
        test_ins_names_set = set()
        for triple in test_triples_set:
            h, t = triple[0], triple[2]
            # 同上
            if h not in ins_ids_dict.keys() or t not in onto_ids_dict.keys():
                continue
            this_head_id, this_tail_id = ins_ids_dict[h], onto_ids_dict[t]
            # 同上
            if h in test_ins_names_set:
                continue
            # 过滤掉已经在训练集中的头部
            if h in train_ins_names_set:
                continue
            test_ins_names_set.add(h)  # 将已经出现过的头放入test_ins_names_set中
            test_heads_ids_list.append(this_head_id)
            test_tails_ids_list.append(this_tail_id)
            test_head_tails_ids_list.append(list(test_head_tails_ids_dict[this_head_id]))
            # 在测试集中test_heads_ids_list[i]对应的实体一定属于test_tails_ids_list[i]对应的这个类型
            assert len(test_heads_ids_list) == len(test_tails_ids_list)
        print("# selected test entity types len:", len(test_heads_ids_list))

        # 注意返回的嵌套列表结构
        return [[train_heads_ids_list, train_tails_ids_list],
                [test_heads_ids_list, test_tails_ids_list, test_head_tails_ids_list]]
    # ----------------------------------------------------------------------------------
    else:
        print("read KG triples...")
        if "insnet" in all_file:
            graph_name = "instance"  # KG中的所有节点表示的实体都是instance
        else:
            graph_name = "ontology"  # KG中的所有节点表示的实体都是ontology
        all_triples_set = read_triples(all_file)
        print("all triples length:", len(all_triples_set))

        # 获得实体set(ents_set)和关系set(rels_set)
        ents_set, rels_set = get_ents_and_rels_set(all_triples_set)
        # 为实体和关系分配id，获得记录各自id的映射字典(ents_ids_dict和rels_ids_dict)
        ents_ids_dict, rels_ids_dict = generate_mapping_id(all_triples_set, ents_set, rels_set)
        # 将三元组的三个部分id化，得到id化后的三元组集合all_ids_triples_set
        all_ids_triples_set = get_ids_triples(all_triples_set, ents_ids_dict, rels_ids_dict)
        # 用all_ids_triples_set初始化自定义的Triples类，更好地管理所需要的三元组的各种变换形式得到all_ids_triples
        all_ids_triples = Triples(all_ids_triples_set)

        total_ents_num = len(all_ids_triples.ents)
        total_rels_num = len(all_ids_triples.rels)
        total_triples_num = len(all_ids_triples.triple_list)
        print("total " + graph_name + " ents num:", total_ents_num)
        print("total " + graph_name + " rels num:", total_rels_num)
        print("total " + graph_name + " triples num:", total_triples_num)

        train_triples_set = read_triples(train_file)
        train_ids_triples_set = get_ids_triples(train_triples_set, ents_ids_dict, rels_ids_dict)

        test_triples_set = read_triples(test_file)
        test_ids_triples_set = get_ids_triples(test_triples_set, ents_ids_dict, rels_ids_dict)

        # 注意返回值的结构
        return [all_ids_triples, train_ids_triples_set, test_ids_triples_set, total_ents_num,
                total_rels_num, total_triples_num], ents_ids_dict


# 从给定的文件中读取(h,r,t)三元组，返回给定文件中所有(h,r,t)三元组组成的set，即all_triples_set
def read_triples(file):
    all_triples_set = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 3
            h = params[0]
            r = params[1]
            t = params[2]
            all_triples_set.add((h, r, t))
        f.close()
    return all_triples_set


# 输入一个triples三元组set(triples_set)
# 输出其实体(包括头实体和尾实体)set和关系set
def get_ents_and_rels_set(triples_set):
    heads_set = set([triple[0] for triple in triples_set])
    tails_set = set([triple[2] for triple in triples_set])
    rels_set = set([triple[1] for triple in triples_set])
    ents_set = heads_set | tails_set
    return ents_set, rels_set


# 输入一个triples_set,记录实体id映射的ents_ids_dict字典和记录关系id映射的rels_ids_dict字典
# 输出id化后的三元组(head_id, rel_id, tail_id)组成的集合ids_triples_set
def get_ids_triples(triples_set, ents_ids_dict, rels_ids_dict):
    ids_triples_set = set()
    for triple in triples_set:
        head, relation, tail = triple[0], triple[1], triple[2]
        ids_triples_set.add((ents_ids_dict[head], rels_ids_dict[relation], ents_ids_dict[tail]))
    assert len(ids_triples_set) == len(triples_set)
    return ids_triples_set


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


# 将triples中ents表示的所有实体和rels表示的所有关系映射为id，返回表示这种映射的dict(ents_ids_dict和rels_ids_dict)
def generate_mapping_id(triples_set, ents_set, rels_set, ordered=True):
    ents_ids_dict = dict()
    rels_ids_dict = dict()

    if ordered:
        ordered_ents_list, ordered_rels_list = sort_elements(triples_set, ents_set, rels_set)
        for i in range(len(ordered_ents_list)):
            ents_ids_dict[ordered_ents_list[i]] = i
        for i in range(len(ordered_rels_list)):
            rels_ids_dict[ordered_rels_list[i]] = i
    else:
        ent_index_cnt = 0
        for ent in ents_set:
            if ent not in ents_ids_dict.keys():
                ents_ids_dict[ent] = ent_index_cnt
                ent_index_cnt += 1
        rel_index_cnt = 0
        for rel in rels_set:
            if rel not in rels_ids_dict.keys():
                rels_ids_dict[rel] = rel_index_cnt
                rel_index_cnt += 1

    assert len(ents_ids_dict) == len(ents_set) and len(rels_ids_dict) == len(rels_set)
    return ents_ids_dict, rels_ids_dict


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
