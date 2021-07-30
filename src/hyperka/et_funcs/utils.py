import numpy as np
import time

from hyperka.et_funcs.triples import Triples


# 输入一个triples
# 输出其头实体、尾实体和关系
def get_ents(triples):
    heads = set([triple[0] for triple in triples])
    tails = set([triple[2] for triple in triples])
    props = set([triple[1] for triple in triples])
    ents = heads | tails
    return ents, props


# 输入一个triples,记录实体id的ent_ids字典和记录关系id的rel_ids字典
# 输出三元组(head_id, rel_id, tail_od)组成的集合
def get_ids_triples(triples, ent_ids, rel_ids):
    ids_triples = set()
    for item in triples:
        ids_triples.add((ent_ids[item[0]], rel_ids[item[1]], ent_ids[item[2]]))
    return ids_triples


# 对elements_set中ents表示的所有实体和props表示的所有关系进行排序
# TODO：具体怎么排序不是很清楚
def sort_elements(triples, elements_set, props):
    dic = dict()
    props_dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in props:
            props_dic[p] = props_dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1

    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]

    props_sorted_list = sorted(props_dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    props_ordered_elements = [x[0] for x in props_sorted_list]

    return ordered_elements, dic, props_ordered_elements, props_dic


# 将triples中ents表示的所有实体和props表示的所有关系映射为各自的id
def generate_mapping_id(triples, ents, props, ordered=True):
    ent_ids = dict()
    rel_ids = dict()

    if ordered:
        ordered_elements, _, props_ordered_elements, _ = sort_elements(triples, ents, props)
        for i in range(len(ordered_elements)):
            ent_ids[ordered_elements[i]] = i
        for i in range(len(props_ordered_elements)):
            rel_ids[props_ordered_elements[i]] = i
    else:
        index = 0
        for ent in ents:
            if ent not in ent_ids:
                ent_ids[ent] = index
                index += 1

        prop_index = 0
        for prop in props:
            if prop not in rel_ids:
                rel_ids[prop] = prop_index
                prop_index += 1

    assert len(ent_ids) == len(set(ents))
    return ent_ids, rel_ids


# 根据参数的要求读取相应的文件
# TODO: 部分代码没看懂
def get_input(all_file, train_file, test_file, if_cross=False, ins_ids=None, onto_ids=None):
    # if_cross表示既有instance又有type
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
        train_ins = set()

        for triple in train_triples:
            # filter the entities that not have triples in the KG
            # 若头实体不是已知的instance或者尾实体不是已知的type则跳过这一个三元组(该三元组不能用于训练)
            if triple[0] not in ins_ids.keys() or triple[2] not in onto_ids.keys():
                continue
            # 头部instance如果重复出现了，也跳过
            # TODO:为什么这里需要跳过？
            if triple[0] in train_ins:
                continue
            train_ins.add(triple[0])
            train_heads_id_list.append(ins_ids[triple[0]])
            train_tails_id_list.append(onto_ids[triple[2]])
            assert len(train_heads_id_list) == len(train_tails_id_list)
        print("# selected train entity types len:", len(train_heads_id_list))

        test_head_tails_id_dict = dict()
        # test_ins = set()
        for triple in test_triples:
            # filter the entities that not have triples in the KG
            # 若头实体不是已知的instance或者尾实体不是已知的type则跳过这一个三元组(该三元组不能用于测试)
            if triple[0] not in ins_ids.keys() or triple[2] not in onto_ids.keys():
                continue
            this_head_id = ins_ids[triple[0]]
            if this_head_id not in test_head_tails_id_dict.keys():
                test_head_tails_id_dict[this_head_id] = set()
            test_head_tails_id_dict[this_head_id].add(onto_ids[triple[2]])
            # test_ins.add(triple[0])

        # print("training & test ins len:", len(train_ins & test_ins))

        # ***************************************
        test_heads_id_list = list()
        test_tails_id_list = list()
        test_head_tails_id_list = list()
        test_ins = set()
        for triple in test_triples:
            # filter the entities that not have triples in the KG
            if triple[0] not in ins_ids.keys() or triple[2] not in onto_ids.keys():
                continue
            if triple[0] in test_ins:
                continue
            # filter the instances in training data
            if triple[0] in train_ins:
                continue
            test_ins.add(triple[0])
            test_heads_id_list.append(ins_ids[triple[0]])
            test_heads_id_list.append(onto_ids[triple[2]])
            test_head_tails_id_list.append(list(test_head_tails_id_dict[ins_ids[triple[0]]]))
            assert len(test_heads_id_list) == len(test_heads_id_list)
            print("# selected test entity types len:", len(test_heads_id_list))

        return [[train_heads_id_list, train_tails_id_list],
                [test_heads_id_list, test_tails_id_list, test_head_tails_id_list]]

    else:
        print("read KG triples...")
        if "insnet" in all_file:
            graph_name = "instance"  # 图中的所有节点表示的实体都是instance
        else:
            graph_name = "ontology"  # 图中的所有节点表示的实体都是type

        triples = read_triples(all_file)
        print("all triples length:", len(triples))

        # 将实体、关系以及三元组全部id化
        ents, props = get_ents(triples)
        ent_ids, rel_ids = generate_mapping_id(triples, ents, props)
        ids_triples = get_ids_triples(triples, ent_ids, rel_ids)
        triples = Triples(ids_triples)

        total_ents_num = len(triples.ents)
        total_props_num = len(triples.props)

        total_triples_num = len(triples.triple_list)
        print("total " + graph_name + " ents num:", total_ents_num)
        print("total " + graph_name + " props num:", total_props_num)
        print("total " + graph_name + " triples num:", total_triples_num)

        train_triples = read_triples(train_file)
        train_ids_triples = get_ids_triples(train_triples, ent_ids, rel_ids)

        test_triples = read_triples(test_file)
        test_ids_triples = get_ids_triples(test_triples, ent_ids, rel_ids)

        return [triples, train_ids_triples, test_ids_triples, total_ents_num,
                total_props_num, total_triples_num], ent_ids


# 读入全部所需数据
def read_input(folder):
    # 使用db数据集
    if "yago" not in folder:
        # insnet 实例和实例的关系
        insnet, ins_ids = get_input(all_file=folder + "db_insnet.txt", train_file=folder + "db_insnet_train.txt",
                                    test_file=folder + "db_insnet_test.txt")
        # onto 类型和类型的关系
        onto, onto_ids = get_input(all_file=folder + "db_onto_small_mini.txt",
                                   train_file=folder + "db_onto_small_train.txt",
                                   test_file=folder + "db_onto_small_test.txt")
        # instype 实例和类型的关系
        instype = get_input(all_file=folder + "db_InsType_mini.txt", train_file=folder + "db_InsType_train.txt",
                            test_file=folder + "db_InsType_test.txt", if_cross=True, ins_ids=ins_ids, onto_ids=onto_ids)
    # 使用yoga数据集
    else:
        # insnet 实例和实例的关系
        insnet, ins_ids = get_input(all_file=folder + "yago_insnet_mini.txt",
                                    train_file=folder + "yago_insnet_train.txt",
                                    test_file=folder + "yago_insnet_test.txt")
        # onto 类型和类型的关系
        onto, onto_ids = get_input(all_file=folder + "yago_ontonet.txt", train_file=folder + "yago_ontonet_train.txt",
                                   test_file=folder + "yago_ontonet_test.txt")
        # instype 实例和类型的关系
        instype = get_input(all_file=folder + "yago_InsType_mini.txt", train_file=folder + "yago_InsType_train.txt",
                            test_file=folder + "yago_InsType_test.txt", if_cross=True, ins_ids=ins_ids,
                            onto_ids=onto_ids)

    return insnet, onto, instype


# 将pairs中的二元组的信息写入file指定的文件中
def pair2file(file, pairs):
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


# 从给定的文件中读取(h,r,t)三元组，返回给定文件中所有(h,r,t)三元组组成的集合
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
