# -*- coding: utf-8 -*-
import os
import math
import time
import random
import hyperka.et_funcs.utils as ut
from hyperka.et_apps.util import generate_adjacent_graph
from hyperka.ea_funcs.train_funcs import find_neighbours_multi

g = 1024 * 1024


# 根据相应参数初始化模型
def get_model(folder, kge_model, params):
    print("data folder:", folder)
    # 用于读取输入的函数
    read_func = ut.read_input
    # insnet和onto的结构如下：
    # [triples, train_ids_triples, test_ids_triples, total_ents_num, total_rels_num, total_triples_num]
    # instype的结构如下：
    # [[train_heads_id_list, train_tails_id_list],[test_heads_id_list, test_tails_id_list, test_head_tails_id_list]]

    print("read_input begin...")
    insnet, onto, instype = read_func(folder)
    print("read_input finished\n")

    print("generate_adjacent_graph begin...")

    ins_adj = generate_adjacent_graph(total_ents_num=insnet[3], triples=insnet[0].triples)
    onto_adj = generate_adjacent_graph(total_ents_num=onto[3], triples=onto[0].triples)

    print("ins_adj shape:", ins_adj[2])
    print("onto_adj shape:", onto_adj[2])
    print("generate_adjacent_graph finished\n")

    model = kge_model(insnet, onto, instype, ins_adj, onto_adj, params)

    return insnet[0], onto[0], model


# TODO:truncated_ins_num和truncated_onto_num的作用不是很明白
# TODO:只知道default设置下开始的时候truncated_ins_num和truncated_onto_num都为0
def train_k_epochs(model, ins_triples, onto_triples, k, params, truncated_ins_num, truncated_onto_num):
    neighbours_of_ins_triples, neighbours_of_onto_triples = dict(), dict()
    start = time.time()
    if truncated_ins_num > 0.1:
        print("this part of codes are not fixed!")
        os.system("pause")

        ins_embeds = model.eval_ins_input_embed()
        onto_embeds = model.eval_onto_input_embed()
        neighbours_of_ins_triples = find_neighbours_multi(ins_embeds, model.ins_entities, truncated_ins_num,
                                                          params.nums_threads)
        neighbours_of_onto_triples = find_neighbours_multi(onto_embeds, model.onto_entities, truncated_onto_num,
                                                           params.nums_threads)
        end = time.time()
        print("generate nearest-{}-&-{} neighbours: {:.3f} s".format(truncated_ins_num, truncated_onto_num,
                                                                     end - start))
    for i in range(k):
        triple_loss, mapping_loss, t2 = train_1_epoch(model, ins_triples, onto_triples, params,
                                                      neighbours_of_ins_triples,
                                                      neighbours_of_onto_triples)
        print("triple_loss(L1) = {:.3f}, typing_loss(L2) = {:.3f}, "
              "time = {:.3f} s".format(triple_loss, mapping_loss, t2))


def train_1_epoch(model, ins_triples, onto_triples, params,
                  neighbours_of_ins_triples, neighbours_of_onto_triples):
    triple_loss, mapping_loss = 0, 0
    start = time.time()
    steps = math.ceil(ins_triples.triples_num / params.batch_size)
    # TODO:link_batch_size的作用未知
    link_batch_size = math.ceil(len(model.train_instype_head) / steps)
    for step in range(steps):
        triple_step_loss, triple_step_end = train_triple_1_step(model, ins_triples, onto_triples, step, params,
                                                                neighbours_of_ins_triples, neighbours_of_onto_triples)
        triple_loss += triple_step_loss
        mapping_step_loss, mapping_step_end = train_mapping_1_step(model, link_batch_size, params.mapping_neg_nums)
        mapping_loss += mapping_step_loss
    triple_loss /= steps
    mapping_loss /= steps
    random.shuffle(ins_triples.triple_list)
    random.shuffle(onto_triples.triple_list)
    end = time.time()
    return triple_loss, mapping_loss, round(end - start, 2)


def train_triple_1_step(model, ins_triples, onto_triples, step, params,
                        neighbours_of_ins_triples, neighbours_of_onto_triples):
    start = time.time()
    # triple_fetches = {"triple_loss": model.triple_loss, "train_triple_optimizer": model.triple_optimizer}

    ins_pos_triples, ins_neg_triples, onto_pos_triples, onto_neg_triples = \
        generate_pos_neg_batch(ins_triples, onto_triples, step, params.batch_size, params.nums_neg,
                               neighbours_of_ins_triples, neighbours_of_onto_triples)

    triple_pos_neg_batch = [ins_pos_triples, ins_neg_triples, onto_pos_triples, onto_neg_triples]
    # triple_feed_dict = {model.ins_pos_h: [x[0] for x in ins_pos_triples],
    #                     model.ins_pos_r: [x[1] for x in ins_pos_triples],
    #                     model.ins_pos_t: [x[2] for x in ins_pos_triples],
    #                     model.ins_neg_h: [x[0] for x in ins_neg_triples],
    #                     model.ins_neg_r: [x[1] for x in ins_neg_triples],
    #                     model.ins_neg_t: [x[2] for x in ins_neg_triples],
    #                     model.onto_pos_h: [x[0] for x in onto_pos_triples],
    #                     model.onto_pos_r: [x[1] for x in onto_pos_triples],
    #                     model.onto_pos_t: [x[2] for x in onto_pos_triples],
    #                     model.onto_neg_h: [x[0] for x in onto_neg_triples],
    #                     model.onto_neg_r: [x[1] for x in onto_neg_triples],
    #                     model.onto_neg_t: [x[2] for x in onto_neg_triples]}

    # TODO:接下来这段代码还没有改好
    triple_loss = model.optimize_triple_loss(triple_pos_neg_batch)
    triple_loss = triple_loss.data
    # results = model.session.run(fetches=triple_fetches, feed_dict=triple_feed_dict)
    # triple_loss = results["triple_loss"]
    end = time.time()
    return triple_loss, round(end - start, 2)


# TODO:这里面构造负例的具体方法还没有理清楚,并且内部变量命名也太混乱，暂且当成黑箱
def train_mapping_1_step(model, link_batch_size, nums_neg=20):
    start = time.time()

    # mapping_fetches = {"mapping_loss": model.mapping_loss, "train_mapping_opt": model.mapping_optimizer}

    pos_link_list = random.sample(model.train_instype_link, link_batch_size)
    link_pos_h = [pos_link[0] for pos_link in pos_link_list]
    link_pos_t = [pos_link[1] for pos_link in pos_link_list]

    neg_link_list = list()
    for i in range(nums_neg):
        neg_tails = random.sample(model.train_instype_tail + model.test_instype_tail, link_batch_size)
        neg_link_list.extend([(link_pos_h[i], neg_tails[i]) for i in range(link_batch_size)])
    neg_link_list = list(set(neg_link_list) - model.train_instype_set)
    link_neg_h = [neg_link[0] for neg_link in neg_link_list]
    link_neg_t = [neg_link[1] for neg_link in neg_link_list]

    mapping_pos_neg_batch = [link_pos_h, link_pos_t, link_neg_h, link_neg_t]
    # feed_dict = {model.cross_pos_left: link_pos_h, model.cross_pos_right: link_pos_t,
    #              model.cross_neg_left: link_neg_h, model.cross_neg_right: link_neg_t}

    # results = model.session.run(fetches=mapping_fetches, feed_dict=feed_dict)
    # mapping_loss = results["mapping_loss"]

    # TODO:接下来这段代码还没有改好
    mapping_loss = model.optimize_mapping_loss(mapping_pos_neg_batch)
    mapping_loss = mapping_loss.data

    end = time.time()
    return mapping_loss, round(end - start, 2)


# 生成正例batch
# TODO:这里具体如何划分,我看代码的时候没怎么认真思考,直接当成黑箱处理
def generate_pos_batch(ins_triples_list, onto_triples_list, step, batch_size):
    num1 = batch_size
    num2 = int(batch_size / len(ins_triples_list) * len(onto_triples_list))
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(ins_triples_list):
        end1 = len(ins_triples_list)
    if end2 > len(onto_triples_list):
        end2 = len(onto_triples_list)
    pos_ins_triples = ins_triples_list[start1: end1]
    pos_onto_triples = onto_triples_list[start2: end2]
    if len(pos_onto_triples) == 0:
        pos_onto_triples = onto_triples_list[0:num2]
    return pos_ins_triples, pos_onto_triples


# 生成负例batch
def generate_neg_triples(pos_triples, all_triples, nums_neg, neighbours):
    all_triples_set = all_triples.triples
    ent_list = all_triples.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            candidates = neighbours.get(h, ent_list)
            neg_samples = random.sample(candidates, nums_neg)
            neg_triples.extend([(neg_head, r, t) for neg_head in neg_samples])
        elif choice >= 500:
            candidates = neighbours.get(t, ent_list)
            neg_samples = random.sample(candidates, nums_neg)
            neg_triples.extend([(h, r, neg_tail) for neg_tail in neg_samples])
    neg_triples = list(set(neg_triples) - all_triples_set)
    return neg_triples


# 这个函数应该是generate_neg_triples_multi()函数的简化版本
# def generate_neg_triples(pos_triples, triples_data):
#     all_triples = triples_data.triples
#     entities = triples_data.ent_list
#     neg_triples = list()
#     for (h, r, t) in pos_triples:
#         h2, r2, t2 = h, r, t
#         while True:
#             choice = random.randint(0, 999)
#             if choice < 500:
#                 h2 = random.sample(entities, 1)[0]
#             elif choice >= 500:
#                 t2 = random.sample(entities, 1)[0]
#             if (h2, r2, t2) not in all_triples:
#                 break
#         neg_triples.append((h2, r2, t2))
#     assert len(neg_triples) == len(pos_triples)
#     return neg_triples

# 生成正例和负例各自的batch

# 生成batch
def generate_pos_neg_batch(ins_triples, onto_triples, step, batch_size, nums_neg,
                           neighbours_of_ins_triples=None, neighbours_of_onto_triples=None):
    assert nums_neg >= 1

    pos_ins_triples, pos_onto_triples = generate_pos_batch(
        ins_triples.triple_list, onto_triples.triple_list, step, batch_size)

    neg_ins_triples = list()
    neg_onto_triples = list()
    neg_ins_triples.extend(
        generate_neg_triples(pos_ins_triples, ins_triples, nums_neg, neighbours_of_ins_triples))
    neg_onto_triples.extend(
        generate_neg_triples(pos_onto_triples, onto_triples, nums_neg, neighbours_of_onto_triples))

    return pos_ins_triples, neg_ins_triples, pos_onto_triples, neg_onto_triples
