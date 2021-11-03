# -*- coding: utf-8 -*-
import math
import os
import random
import time

import src.hyperka.et_funcs.utils as ut
from src.hyperka.ea_funcs.train_funcs import find_neighbours_multi
from src.hyperka.et_apps.util import generate_adjacent_graph


# 根据相应参数初始化模型
def get_model(folder, kge_model, args):
    print("data folder:", folder)

    print("read_input begin...")
    read_func = ut.read_input  # 用于读取输入的函数
    # insnet和onto的结构如下：
    # [all_ids_triples, train_ids_triples_set, test_ids_triples_set,
    # total_ents_num, total_rels_num, total_triples_num]
    # instype的结构如下：
    # [[train_heads_ids_list, train_tails_ids_list],
    # [test_heads_ids_list, test_tails_ids_list, test_head_tails_ids_list]]
    insnet, onto, instype = read_func(folder)
    print("read_input finished\n")

    print("generate_adjacent_graph begin...")
    ins_near_ents_graph, ins_near_rels_graph = generate_adjacent_graph(total_ents_num=insnet[3],
                                                                       total_rels_num=insnet[4],
                                                                       triples=insnet[0].triples)
    onto_near_ents_graph, onto_near_rels_graph = generate_adjacent_graph(total_ents_num=onto[3],
                                                                         total_rels_num=onto[4],
                                                                         triples=onto[0].triples)
    print("ins_near_ents_adj shape:", ins_near_ents_graph[0].shape)
    print("ins_near_rels_adj shape:", ins_near_rels_graph[0].shape)
    print("onto_near_ents_adj shape:", onto_near_ents_graph[0].shape)
    print("onto_near_rels_adj shape:", onto_near_rels_graph[0].shape)
    print("generate_adjacent_graph finished\n")

    model = kge_model(insnet, onto, instype, ins_near_ents_graph, ins_near_rels_graph, onto_near_ents_graph,
                      onto_near_rels_graph, args)

    return insnet[0], onto[0], model


# 获得修改前的HyperKA模型
def get_origin_model(folder, kge_model, args):
    print("data folder:", folder)

    print("read_input begin...")
    read_func = ut.read_input  # 用于读取输入的函数
    # insnet和onto的结构如下：
    # [all_ids_triples, train_ids_triples_set, test_ids_triples_set,
    # total_ents_num, total_rels_num, total_triples_num]
    # instype的结构如下：
    # [[train_heads_ids_list, train_tails_ids_list],
    # [test_heads_ids_list, test_tails_ids_list, test_head_tails_ids_list]]
    insnet, onto, instype = read_func(folder)
    print("read_input finished\n")

    print("generate_adjacent_graph begin...")
    ins_adj = generate_adjacent_graph(total_ents_num=insnet[3], total_rels_num=insnet[4], triples=insnet[0].triples,
                                      origin=True)
    onto_adj = generate_adjacent_graph(total_ents_num=onto[3], total_rels_num=onto[4], triples=onto[0].triples,
                                       origin=True)
    print("ins adj shape:", ins_adj.shape)
    print("onto adj shape:", ins_adj.shape)

    model = kge_model(insnet, onto, instype, ins_adj, onto_adj, args)

    return insnet[0], onto[0], model


# 训练k个epoch
def train_k_epochs(model, ins_triples, onto_triples, k, args, truncated_ins_num, truncated_onto_num):
    neighbours_of_ins_triples, neighbours_of_onto_triples = dict(), dict()
    start = time.time()
    # TODO:truncated_ins_num和truncated_onto_num的作用不是很明白，default设置下开始的时候truncated_ins_num和truncated_onto_num都为0
    if truncated_ins_num > 0.1:
        ins_embeds = model.eval_ins_input_embed()
        onto_embeds = model.eval_onto_input_embed()
        neighbours_of_ins_triples = find_neighbours_multi(ins_embeds, model.ins_entities, truncated_ins_num,
                                                          args.nums_threads)
        neighbours_of_onto_triples = find_neighbours_multi(onto_embeds, model.onto_entities, truncated_onto_num,
                                                           args.nums_threads)
        end = time.time()
        print("generate nearest-{}-&-{} neighbours: {:.3f} s".format(truncated_ins_num, truncated_onto_num,
                                                                     end - start))
    for epoch in range(1, k + 1):
        print("epoch:", epoch)
        triple_loss, mapping_loss, time_cost = train_1_epoch(model, ins_triples, onto_triples, args,
                                                             neighbours_of_ins_triples,
                                                             neighbours_of_onto_triples)
        print("triple_loss(L1) = {:.3f}, mapping_loss(L2) = {:.3f}, "
              "time = {:.3f} s".format(triple_loss, mapping_loss, time_cost))

    end = time.time()
    print("train k epochs finished, time cost:", round(end - start, 2), "s")


# 训练1个epoch
def train_1_epoch(model, ins_triples, onto_triples, args,
                  neighbours_of_ins_triples, neighbours_of_onto_triples):
    triple_loss, mapping_loss = 0, 0
    start = time.time()
    # 一个epoch需要跑steps步，每一步跑batch_size大小的数据
    steps = math.ceil(ins_triples.triples_num / args.batch_size)
    # print("steps per epoch:", steps)
    link_batch_size = math.ceil(len(model.train_instype_head) / steps)
    for step in range(1, steps + 1):
        # if step % 5 == 1:
        #     print("\tstep:", step)
        triple_step_loss, triple_step_time = train_triple_1_step(model, ins_triples, onto_triples, step, args,
                                                                 neighbours_of_ins_triples, neighbours_of_onto_triples)
        triple_loss += triple_step_loss
        mapping_step_loss, mapping_step_time = train_mapping_1_step(model, link_batch_size, args.mapping_neg_nums)
        mapping_loss += mapping_step_loss
        # print("train triple 1 step time cost:", triple_step_time, "s")
        # print("train mapping 1 step time cost:", mapping_step_time, "s")
    triple_loss /= steps
    mapping_loss /= steps
    # 一个epoch跑完后对ins_triples_list和onto_triples_list重新排列，这样使下一个epoch时构造的batch与这个epoch的不同
    random.shuffle(ins_triples.triple_list)
    random.shuffle(onto_triples.triple_list)
    end = time.time()
    return triple_loss, mapping_loss, round(end - start, 2)


# 根据triple loss训练一步
def train_triple_1_step(model, ins_triples, onto_triples, step, args,
                        neighbours_of_ins_triples, neighbours_of_onto_triples):
    start = time.time()

    ins_pos_triples, ins_neg_triples, onto_pos_triples, onto_neg_triples = \
        generate_pos_neg_triple_batch(ins_triples, onto_triples, step, args.batch_size, args.triple_neg_nums,
                                      neighbours_of_ins_triples, neighbours_of_onto_triples)

    triple_pos_neg_batch = [ins_pos_triples, ins_neg_triples, onto_pos_triples, onto_neg_triples]

    triple_loss = model.optimize_triple_loss(triple_pos_neg_batch).data

    end = time.time()

    return triple_loss, round(end - start, 2)


# 根据mapping loss训练一步
# TODO:这里面构造负例的具体方法还没有理清楚,并且内部变量命名也比较混乱，暂且命名做出一定修改当成黑箱
def train_mapping_1_step(model, link_batch_size, mapping_neg_nums=20):
    start = time.time()
    # 从model.train_instype_link中选择link_batch_size个实例-类型二元组(ent, type)来作为pos_link_list
    pos_link_list = random.sample(model.train_instype_link, link_batch_size)
    link_pos_h = [pos_link[0] for pos_link in pos_link_list]
    link_pos_t = [pos_link[1] for pos_link in pos_link_list]

    neg_link_list = list()
    for i in range(mapping_neg_nums):
        # 随机选择model中所有type中的link_batch_size个组成neg_tails
        neg_tails = random.sample(model.train_instype_tail + model.test_instype_tail, link_batch_size)
        neg_link_list.extend([(link_pos_h[i], neg_tails[i]) for i in range(link_batch_size)])
    neg_link_list = list(set(neg_link_list) - model.train_instype_set)
    link_neg_h = [neg_link[0] for neg_link in neg_link_list]
    link_neg_t = [neg_link[1] for neg_link in neg_link_list]

    mapping_pos_neg_batch = [link_pos_h, link_pos_t, link_neg_h, link_neg_t]

    mapping_loss = model.optimize_mapping_loss(mapping_pos_neg_batch).data

    end = time.time()
    return mapping_loss, round(end - start, 2)


# 生成某一个epoch的第step步的pos triples
def generate_pos_triples(ins_triples_list, onto_triples_list, step, batch_size):
    # 每一个step需要num1个ins_triples_list中的正例triples和num2个onto_triples_list中的正例triples
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
    # TODO: 这里的特判的意图不是很清楚
    if len(pos_onto_triples) == 0:
        pos_onto_triples = onto_triples_list[0:num2]
    return pos_ins_triples, pos_onto_triples


# 生成某一个epoch的第step步的neg triples
def generate_neg_triples(pos_triples, all_triples, triple_neg_nums, neighbours):
    all_triples_set = all_triples.triples
    ent_list = all_triples.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            # 对每一个pos_triple中的pos triple(设为(h,r,t))，
            # 从pos triple的头部h的邻居中中选取triple_neg_nums个实体来作负例的neg_head
            # 若h的邻居为空，则默认从所有实体(即ent_list)中选择
            candidates = neighbours.get(h, ent_list)
            neg_samples = random.sample(candidates, triple_neg_nums)
            neg_triples.extend([(neg_head, r, t) for neg_head in neg_samples])
        elif choice >= 500:
            # 对每一个pos_triple中的pos triple(设为(h,r,t))，
            # 从pos triple的尾部t的邻居中中选取triple_neg_nums个实体来作负例的neg_tail
            # 若t的邻居为空，则默认从所有实体(即ent_list)中选择
            candidates = neighbours.get(t, ent_list)
            neg_samples = random.sample(candidates, triple_neg_nums)
            neg_triples.extend([(h, r, neg_tail) for neg_tail in neg_samples])
    neg_triples = list(set(neg_triples) - all_triples_set)
    return neg_triples


# 生成某一个epoch的第step步的triple batch(包含neg负例和pos正例)
def generate_pos_neg_triple_batch(ins_triples, onto_triples, step, batch_size, triple_neg_nums,
                                  neighbours_of_ins_triples=None, neighbours_of_onto_triples=None):
    assert triple_neg_nums >= 1

    pos_ins_triples, pos_onto_triples = generate_pos_triples(
        ins_triples.triple_list, onto_triples.triple_list, step, batch_size)

    neg_ins_triples = generate_neg_triples(pos_ins_triples, ins_triples, triple_neg_nums, neighbours_of_ins_triples)
    neg_onto_triples = generate_neg_triples(pos_onto_triples, onto_triples, triple_neg_nums, neighbours_of_onto_triples)

    return pos_ins_triples, neg_ins_triples, pos_onto_triples, neg_onto_triples
