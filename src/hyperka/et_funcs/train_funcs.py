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
    # [[train_heads_id_list, train_tails_id_list],[test_heads_id_list, test_tails_id_list, test_head_tails_id_
    print("read_input begin...")
    insnet, onto, instype = read_func(folder)
    print("read_input finished\n")
    os.system("pause")

    print("generate_adjacent_graph begin...")
    ins_adj = generate_adjacent_graph(total_ents_num=insnet[3], triples=insnet[0].triples)
    onto_adj = generate_adjacent_graph(total_ents_num=onto[3], triples=onto[0].triples)
    print("generate_adjacent_graph finished\n")
    os.system("pause")  # 2021/8/17今天下午重新看代码看到了这里(可以合法正确运行到此处)

    model = kge_model(insnet, onto, instype, ins_adj, onto_adj, params)

    return insnet[0], onto[0], model


# trunc_num的作用不是很明白
def train_k_epochs(model, ins_triples, onto_triples, k, params, trunc_num1, trunc_num2):
    neighbours_4_ins_triples, neighbours_4_onto_triples = dict(), dict()
    t1 = time.time()
    if trunc_num1 > 0.1:
        ins_embeds = model.eval_ins_input_embed()
        onto_embeds = model.eval_onto_input_embed()
        neighbours_4_ins_triples = find_neighbours_multi(ins_embeds, model.ins_entities, trunc_num1,
                                                         params.nums_threads)
        neighbours_4_onto_triples = find_neighbours_multi(onto_embeds, model.onto_entities, trunc_num2,
                                                          params.nums_threads)
        print("generate nearest-{}-&-{} neighbours: {:.3f} s".format(trunc_num1, trunc_num2, time.time() - t1))
    for i in range(k):
        loss1, loss2, t2 = train_1_epoch(model, ins_triples, onto_triples, params, neighbours_4_ins_triples,
                                         neighbours_4_onto_triples)
        print("triple_loss(L1) = {:.3f}, typing_loss(L2) = {:.3f}, time = {:.3f} s".format(loss1, loss2, t2))


def train_1_epoch(model, ins_triples, onto_triples, params, neighbours1, neighbours2):
    triple_loss = 0
    mapping_loss = 0
    start = time.time()
    steps = math.ceil(ins_triples.triples_num / params.batch_size)
    link_batch_size = math.ceil(len(model.seed_sup_ent1) / steps)
    for step in range(steps):
        loss1, t1 = train_triple_1_step(model, ins_triples, onto_triples, step, params, neighbours1, neighbours2)
        triple_loss += loss1
        loss2, t2 = train_mapping_1_step(model, link_batch_size, params.mapping_neg_nums)
        mapping_loss += loss2
    triple_loss /= steps
    mapping_loss /= steps
    random.shuffle(ins_triples.triple_list)
    random.shuffle(onto_triples.triple_list)
    end = time.time()
    return triple_loss, mapping_loss, round(end - start, 2)


def train_triple_1_step(model, ins_triples, onto_triples, step, params, neighbours1, neighbours2):
    start = time.time()
    triple_fetches = {"triple_loss": model.triple_loss, "train_triple_opt": model.triple_optimizer}
    ins_pos, ins_neg, onto_pos, onto_neg = generate_pos_neg_batch(ins_triples, onto_triples, step,
                                                                  params.batch_size, multi=params.nums_neg,
                                                                  neighbours1=neighbours1, neighbours2=neighbours2)

    triple_feed_dict = {model.ins_pos_h: [x[0] for x in ins_pos],
                        model.ins_pos_r: [x[1] for x in ins_pos],
                        model.ins_pos_t: [x[2] for x in ins_pos],
                        model.ins_neg_h: [x[0] for x in ins_neg],
                        model.ins_neg_r: [x[1] for x in ins_neg],
                        model.ins_neg_t: [x[2] for x in ins_neg],
                        model.onto_pos_h: [x[0] for x in onto_pos],
                        model.onto_pos_r: [x[1] for x in onto_pos],
                        model.onto_pos_t: [x[2] for x in onto_pos],
                        model.onto_neg_h: [x[0] for x in onto_neg],
                        model.onto_neg_r: [x[1] for x in onto_neg],
                        model.onto_neg_t: [x[2] for x in onto_neg]}
    results = model.session.run(fetches=triple_fetches, feed_dict=triple_feed_dict)
    triple_loss = results["triple_loss"]
    end = time.time()
    return triple_loss, round(end - start, 2)


# 这里面构造负例的具体方法还没有理清楚
def train_mapping_1_step(model, link_batch_size, multi=20):
    start = time.time()
    mapping_fetches = {"mapping_loss": model.mapping_loss, "train_mapping_opt": model.mapping_optimizer}
    pos_list = random.sample(model.seed_links, link_batch_size)
    pos_entities1 = [pos_link[0] for pos_link in pos_list]
    pos_entities2 = [pos_link[1] for pos_link in pos_list]

    neg_links = list()
    for i in range(multi):
        neg_ent2 = random.sample(model.seed_sup_ent2 + model.ref_ent2, link_batch_size)
        neg_links.extend([(pos_entities1[i], neg_ent2[i]) for i in range(link_batch_size)])
    neg_links = list(set(neg_links) - model.seed_link_set)

    neg_entities1 = [neg_link[0] for neg_link in neg_links]
    neg_entities2 = [neg_link[1] for neg_link in neg_links]
    feed_dict = {model.cross_pos_left: pos_entities1, model.cross_pos_right: pos_entities2,
                 model.cross_neg_left: neg_entities1, model.cross_neg_right: neg_entities2}
    results = model.session.run(fetches=mapping_fetches, feed_dict=feed_dict)
    mapping_loss = results["mapping_loss"]
    end = time.time()
    return mapping_loss, round(end - start, 2)


# 这里的划分有一些边界情况，我看代码的时候没怎么认真思考
def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = batch_size
    num2 = int(batch_size / len(triples1) * len(triples2))
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    if len(pos_triples2) == 0:
        pos_triples2 = triples2[0:num2]
    return pos_triples1, pos_triples2


def generate_neg_triples_multi(pos_triples, triples_data, multi, neighbours):
    all_triples = triples_data.triples
    entities = triples_data.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            candidates = neighbours.get(h, entities)
            neg_samples = random.sample(candidates, multi)
            neg_triples.extend([(neg_head, r, t) for neg_head in neg_samples])
        elif choice >= 500:
            candidates = neighbours.get(t, entities)
            neg_samples = random.sample(candidates, multi)
            neg_triples.extend([(h, r, neg_tail) for neg_tail in neg_samples])
    neg_triples = list(set(neg_triples) - all_triples)
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


def generate_pos_neg_batch(triples1, triples2, step, batch_size, multi=1, neighbours1=None, neighbours2=None):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples1 = list()
    neg_triples2 = list()

    # multi2 = multi
    # multi2 = math.ceil(multi / len(pos_triples1) * len(pos_triples2))
    neg_triples1.extend(generate_neg_triples_multi(pos_triples1, triples1, multi, neighbours1))
    neg_triples2.extend(generate_neg_triples_multi(pos_triples2, triples2, multi, neighbours2))

    return pos_triples1, neg_triples1, pos_triples2, neg_triples2
